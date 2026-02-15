"""Deduplicate CaImAn ROI components based on spatial-footprint similarity.

Problem this addresses:
Sometimes CNMF produces multiple components for the same neuron.

Important note (why your cosine-based test showed max ~0.38):
CaImAn often stores spatial footprints that are L2-normalized per component.
If two components correspond to the same neuron but have slightly shifted or
partially split footprints, their raw cosine overlap can be modest even though
they are duplicates visually.

Therefore, this script supports two criteria (used as OR):
    1) Spatial cosine similarity >= --cos_thr (classic “same footprint”)
    2) Centroid distance <= --max_dist AND temporal correlation >= --temporal_thr
         (classic “same neuron activity in same place”)

Inputs (in a results folder):
  - A_spatial_components.npz  (pixels x n_cells) sparse CSC/CSR
  - F_dff.npy                 (n_cells x T) optional but used for scoring reps
  - S_deconv.npy              (n_cells x T) optional

Outputs (written next to inputs):
  - A_spatial_components_dedup.npz
  - F_dff_dedup.npy (if F_dff.npy exists)
  - S_deconv_dedup.npy (if S_deconv.npy exists)
  - dedup_map.json (cluster + index mapping)

Usage:
    python deduplicate_rois.py --results results/well1-003 --cos_thr 0.95
    python deduplicate_rois.py --results results/well1-003 --max_dist 10 --temporal_thr 0.95
  python deduplicate_rois.py --results results/well1-003 --dry_run
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz, csc_matrix
from scipy.spatial import cKDTree

try:
    import caiman as cm  # type: ignore
except Exception:  # pragma: no cover
    cm = None


@dataclass
class DedupResult:
    kept_indices: list[int]
    dropped_indices: list[int]
    groups: list[list[int]]
    representative_for_group: list[int]


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def compute_cosine_similarity(A: csc_matrix) -> np.ndarray:
    """Return dense (n_cells x n_cells) cosine similarity matrix."""
    # Gram matrix: (n x n)
    G = (A.T @ A).toarray().astype(np.float64, copy=False)
    norms = np.sqrt(np.clip(np.diag(G), 0.0, None))
    denom = norms[:, None] * norms[None, :] + 1e-12
    C = G / denom
    np.fill_diagonal(C, 1.0)
    return C


def find_memmap(results: Path, mmap_arg: str | None) -> Path | None:
    if mmap_arg:
        p = Path(mmap_arg).expanduser().resolve()
        return p if p.exists() else None
    candidates = sorted(results.glob("mc_Corder*.mmap"))
    if not candidates:
        candidates = sorted(results.glob("*.mmap"))
    return candidates[0] if candidates else None


def load_memmap_dims(memmap_path: Path) -> tuple[int, int]:
    """Return (d1, d2) from a CaImAn memmap.

    Falls back to parsing from the filename if possible.
    """
    m = re.search(r"d1_(\d+)_d2_(\d+)", memmap_path.name)
    if m:
        return int(m.group(1)), int(m.group(2))

    if cm is None:
        raise RuntimeError("CaImAn is not available to read memmap dims, and dims were not parseable from filename.")

    # Yr: (pixels, T), dims: (d1, d2), T: frames
    _, dims, _ = cm.load_memmap(str(memmap_path))
    return int(dims[0]), int(dims[1])


def centroids_from_A(A: csc_matrix, dims: tuple[int, int]) -> np.ndarray:
    """Compute weighted centroids (y, x) for each component column in A.

    Uses Fortran ordering (CaImAn convention): pixel index -> (y, x) via order='F'.
    Returns array shape (n_cells, 2) with NaNs for empty components.
    """
    d1, d2 = int(dims[0]), int(dims[1])
    n_cells = A.shape[1]
    cent = np.full((n_cells, 2), np.nan, dtype=float)

    A_csc = A.tocsc()
    for j in range(n_cells):
        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        if start == end:
            continue
        idx = A_csc.indices[start:end]
        w = A_csc.data[start:end].astype(float, copy=False)
        wsum = float(np.sum(w))
        if wsum <= 0:
            continue
        ys, xs = np.unravel_index(idx, (d1, d2), order="F")
        cent[j, 0] = float(np.sum(ys * w) / wsum)
        cent[j, 1] = float(np.sum(xs * w) / wsum)
    return cent


def core_masks_from_A(A: csc_matrix, mask_thr: float) -> tuple[list[np.ndarray], np.ndarray]:
    """Return per-component core pixel indices and their sizes.

    Core mask is defined as pixels where a >= mask_thr * max(a) for that component.
    Uses sparse columns and returns indices (sorted) without densifying.
    """
    if not (0.0 < float(mask_thr) <= 1.0):
        raise ValueError("mask_thr must be in (0, 1].")

    A_csc = A.tocsc()
    n_cells = A_csc.shape[1]
    cores: list[np.ndarray] = []
    sizes = np.zeros(n_cells, dtype=int)
    for j in range(n_cells):
        start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
        if start == end:
            cores.append(np.array([], dtype=np.int64))
            continue
        data = A_csc.data[start:end]
        idx = A_csc.indices[start:end]
        mx = float(np.max(data))
        if mx <= 0:
            cores.append(np.array([], dtype=np.int64))
            continue
        thr = float(mask_thr) * mx
        sel = data >= thr
        core_idx = idx[sel]
        cores.append(core_idx.astype(np.int64, copy=False))
        sizes[j] = int(core_idx.size)
    return cores, sizes


def iou_and_overlap(a_idx: np.ndarray, b_idx: np.ndarray) -> tuple[float, float]:
    """Compute IoU and overlap fraction between two 1D index sets."""
    if a_idx.size == 0 or b_idx.size == 0:
        return 0.0, 0.0
    inter = np.intersect1d(a_idx, b_idx, assume_unique=False).size
    if inter == 0:
        return 0.0, 0.0
    union = int(a_idx.size + b_idx.size - inter)
    iou = float(inter) / float(union) if union > 0 else 0.0
    overlap = float(inter) / float(min(a_idx.size, b_idx.size))
    return iou, overlap


def core_trees_from_indices(cores: list[np.ndarray], dims: tuple[int, int]) -> list[cKDTree | None]:
    d1, d2 = int(dims[0]), int(dims[1])
    trees: list[cKDTree | None] = []
    for idx in cores:
        if idx.size == 0:
            trees.append(None)
            continue
        ys, xs = np.unravel_index(idx, (d1, d2), order="F")
        pts = np.stack([ys.astype(float, copy=False), xs.astype(float, copy=False)], axis=1)
        trees.append(cKDTree(pts))
    return trees


def min_core_pixel_dist(tree_a: cKDTree | None, tree_b: cKDTree | None) -> float:
    if tree_a is None or tree_b is None:
        return float("inf")
    # Query smaller -> larger for speed
    if tree_a.n > tree_b.n:
        tree_a, tree_b = tree_b, tree_a
    d, _ = tree_b.query(tree_a.data, k=1)
    return float(np.min(d)) if np.size(d) else float("inf")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size == 0:
        return float("nan")
    a0 = a - np.nanmean(a)
    b0 = b - np.nanmean(b)
    da = float(np.nanstd(a0))
    db = float(np.nanstd(b0))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.nanmean((a0 / da) * (b0 / db)))


def report_pair_stats(
    *,
    A: csc_matrix,
    F_dff: np.ndarray | None,
    dims: tuple[int, int] | None,
    cos_thr: float | None,
    mask_thr: float,
    max_dist: float,
    pixel_dist_thr: float,
    top_k: int,
) -> None:
    if dims is None:
        print("[report] dims unavailable; cannot compute centroid/mask overlap stats")
        return

    n_cells = A.shape[1]
    cent = centroids_from_A(A, dims)
    cores, _ = core_masks_from_A(A, mask_thr=mask_thr)
    trees = core_trees_from_indices(cores, dims)

    C = compute_cosine_similarity(A) if cos_thr is not None else None

    pairs: list[tuple[float, float, float, float, float, float, int, int]] = []
    # fields: corr, dist, iou, overlap, minpix, cosine, i, j
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            if np.any(np.isnan(cent[i])) or np.any(np.isnan(cent[j])):
                continue
            dy = float(cent[i, 0] - cent[j, 0])
            dx = float(cent[i, 1] - cent[j, 1])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > float(max_dist):
                continue
            iou, ov = iou_and_overlap(cores[i], cores[j])
            md = min_core_pixel_dist(trees[i], trees[j])
            cosv = float(C[i, j]) if C is not None else float("nan")
            corr = safe_corr(F_dff[i], F_dff[j]) if F_dff is not None else float("nan")
            pairs.append((corr, dist, iou, ov, md, cosv, i, j))

    if not pairs:
        print(f"[report] No pairs within max_dist={max_dist}")
        return

    corr_vals = np.array([p[0] for p in pairs if np.isfinite(p[0])], dtype=float)
    iou_vals = np.array([p[2] for p in pairs if np.isfinite(p[2])], dtype=float)
    ov_vals = np.array([p[3] for p in pairs if np.isfinite(p[3])], dtype=float)
    md_vals = np.array([p[4] for p in pairs if np.isfinite(p[4])], dtype=float)
    cos_vals = np.array([p[5] for p in pairs if np.isfinite(p[5])], dtype=float)

    def q(x: np.ndarray, p: float) -> float:
        return float(np.quantile(x, p)) if x.size else float("nan")

    print(f"[report] Pairs considered (within {max_dist}px): {len(pairs)}")
    if corr_vals.size:
        print(f"[report] corr: max={corr_vals.max():0.3f}  p95={q(corr_vals,0.95):0.3f}  p90={q(corr_vals,0.90):0.3f}")
    print(f"[report] iou:  max={iou_vals.max():0.3f}  p95={q(iou_vals,0.95):0.3f}  p90={q(iou_vals,0.90):0.3f}")
    print(f"[report] ov:   max={ov_vals.max():0.3f}  p95={q(ov_vals,0.95):0.3f}  p90={q(ov_vals,0.90):0.3f}")
    print(
        f"[report] minpix: min={md_vals.min():0.2f}  p05={q(md_vals,0.05):0.2f}  p10={q(md_vals,0.10):0.2f}  "
        f"(<= {pixel_dist_thr} indicates touching/adjacent cores)"
    )
    if cos_vals.size:
        print(f"[report] cos:  max={cos_vals.max():0.3f}  p95={q(cos_vals,0.95):0.3f}  p90={q(cos_vals,0.90):0.3f}")

    # Top pairs by each metric
    def show(title: str, key_idx: int, reverse: bool = True) -> None:
        def key(t):
            v = t[key_idx]
            if not np.isfinite(v):
                return -1e9 if reverse else 1e9
            return v

        pairs_sorted = sorted(pairs, key=key, reverse=reverse)
        print(f"\n[report] Top {top_k} by {title} (within {max_dist}px):")
        for corr, dist, iou, ov, md, cosv, i, j in pairs_sorted[:top_k]:
            print(f"  pair=({i},{j}) dist={dist:0.2f} corr={corr:0.3f} iou={iou:0.3f} ov={ov:0.3f} minpix={md:0.2f} cos={cosv:0.3f}")

    if top_k > 0:
        show("corr", 0)
        show("iou", 2)
        show("ov", 3)
        show("minpix", 4, reverse=False)
        if C is not None:
            show("cos", 5)


def pick_representative(indices: list[int], F_dff: np.ndarray | None, norms: np.ndarray) -> int:
    """Pick best component in a duplicate group."""
    if F_dff is not None and F_dff.size > 0:
        # higher variance => stronger / cleaner signal, usually
        vars_ = [float(np.nanvar(F_dff[i])) for i in indices]
        return indices[int(np.argmax(vars_))]
    # fallback: keep largest spatial norm
    vals = [float(norms[i]) for i in indices]
    return indices[int(np.argmax(vals))]


def deduplicate(
    A: csc_matrix,
    *,
    cos_thr: float | None,
    F_dff: np.ndarray | None,
    dims: tuple[int, int] | None,
    max_dist: float,
    temporal_thr: float,
    mask_thr: float,
    iou_thr: float,
    overlap_thr: float,
    pixel_dist_thr: float,
) -> DedupResult:
    n_cells = A.shape[1]
    if n_cells <= 1:
        return DedupResult(kept_indices=list(range(n_cells)), dropped_indices=[], groups=[], representative_for_group=[])

    # precompute norms from diagonal of Gram
    norms = np.sqrt(np.clip(np.diag((A.T @ A).toarray()), 0.0, None))

    C = compute_cosine_similarity(A) if cos_thr is not None else None
    cent = centroids_from_A(A, dims) if dims is not None else None
    cores, core_sizes = core_masks_from_A(A, mask_thr=mask_thr) if dims is not None else (None, None)
    trees = core_trees_from_indices(cores, dims) if (dims is not None and cores is not None) else None

    uf = UnionFind(n_cells)
    # Build edges for duplicates; only need upper triangle
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            merged = False
            if C is not None and cos_thr is not None and C[i, j] >= cos_thr:
                uf.union(i, j)
                merged = True

            # If cosine didn't merge them (or cosine disabled), try core-mask overlap.
            # This catches partial/shifted footprints that still belong to the same soma.
            if not merged and cent is not None and cores is not None and core_sizes is not None:
                if core_sizes[i] == 0 or core_sizes[j] == 0:
                    pass
                else:
                    if not (np.any(np.isnan(cent[i])) or np.any(np.isnan(cent[j]))):
                        dy = float(cent[i, 0] - cent[j, 0])
                        dx = float(cent[i, 1] - cent[j, 1])
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist <= float(max_dist):
                            iou, ov = iou_and_overlap(cores[i], cores[j])
                            close_pixels = False
                            if trees is not None:
                                md = min_core_pixel_dist(trees[i], trees[j])
                                close_pixels = md <= float(pixel_dist_thr)
                            if iou >= float(iou_thr) or ov >= float(overlap_thr) or close_pixels:
                                uf.union(i, j)
                                merged = True

            # If still not merged, try spatial+temporal correlation (if traces available)
            if not merged and cent is not None and F_dff is not None:
                if np.any(np.isnan(cent[i])) or np.any(np.isnan(cent[j])):
                    continue
                dy = float(cent[i, 0] - cent[j, 0])
                dx = float(cent[i, 1] - cent[j, 1])
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > float(max_dist):
                    continue
                corr = safe_corr(F_dff[i], F_dff[j])
                if corr >= float(temporal_thr):
                    uf.union(i, j)

    # Collect connected components
    comp: dict[int, list[int]] = {}
    for i in range(n_cells):
        r = uf.find(i)
        comp.setdefault(r, []).append(i)

    groups = [sorted(v) for v in comp.values() if len(v) > 1]
    groups.sort(key=lambda g: g[0])

    dropped: set[int] = set()
    reps: list[int] = []
    for g in groups:
        rep = pick_representative(g, F_dff=F_dff, norms=norms)
        reps.append(rep)
        for idx in g:
            if idx != rep:
                dropped.add(idx)

    kept = [i for i in range(n_cells) if i not in dropped]
    return DedupResult(
        kept_indices=kept,
        dropped_indices=sorted(dropped),
        groups=groups,
        representative_for_group=reps,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Results folder containing A_spatial_components.npz")
    ap.add_argument(
        "--cos_thr",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for considering ROIs duplicates (A overlap).",
    )
    ap.add_argument(
        "--mask_thr",
        type=float,
        default=0.2,
        help="Core mask threshold fraction: keep pixels where a >= mask_thr * max(a).",
    )
    ap.add_argument(
        "--iou_thr",
        type=float,
        default=0.15,
        help="Min IoU (Jaccard) between core masks to consider duplicates.",
    )
    ap.add_argument(
        "--overlap_thr",
        type=float,
        default=0.4,
        help="Min overlap fraction between core masks to consider duplicates (intersection/min(size)).",
    )
    ap.add_argument(
        "--pixel_dist_thr",
        type=float,
        default=3.0,
        help="Max min pixel distance between core masks to consider duplicates (captures split/adjacent ROIs).",
    )
    ap.add_argument(
        "--max_dist",
        type=float,
        default=12.0,
        help="Max centroid distance (pixels) for temporal-based deduplication.",
    )
    ap.add_argument(
        "--temporal_thr",
        type=float,
        default=0.95,
        help="Min correlation between traces (F_dff) for temporal-based deduplication.",
    )
    ap.add_argument(
        "--mmap",
        default=None,
        help="Path to .mmap file (optional). If omitted, auto-detect inside results folder.",
    )
    ap.add_argument(
        "--report_top",
        type=int,
        default=0,
        help="Print diagnostic top pairs within --max_dist by corr/IoU/overlap (0 disables).",
    )
    ap.add_argument("--dry_run", action="store_true", help="Only report duplicates; do not write *_dedup files")
    args = ap.parse_args()

    results = Path(args.results).expanduser().resolve()
    A_path = results / "A_spatial_components.npz"
    if not A_path.exists():
        raise SystemExit(f"Missing: {A_path}")

    F_path = results / "F_dff.npy"
    S_path = results / "S_deconv.npy"

    A = load_npz(str(A_path)).tocsc()
    F_dff = np.load(str(F_path)) if F_path.exists() else None

    # Try to get dims for centroids (needed for temporal-based dedup)
    dims = None
    memmap = find_memmap(results, args.mmap)
    if memmap is not None and memmap.exists():
        try:
            dims = load_memmap_dims(memmap)
        except Exception:
            dims = None

    if dims is None:
        print("Note: could not determine image dims (needed for centroid/mask methods).")
    if F_dff is None:
        print("Note: F_dff.npy not found; temporal correlation method disabled.")

    if int(args.report_top) > 0:
        report_pair_stats(
            A=A,
            F_dff=F_dff,
            dims=dims,
            cos_thr=float(args.cos_thr) if float(args.cos_thr) > 0 else None,
            mask_thr=float(args.mask_thr),
            max_dist=float(args.max_dist),
            pixel_dist_thr=float(args.pixel_dist_thr),
            top_k=int(args.report_top),
        )

    res = deduplicate(
        A,
        cos_thr=float(args.cos_thr) if float(args.cos_thr) > 0 else None,
        F_dff=F_dff,
        dims=dims,
        max_dist=float(args.max_dist),
        temporal_thr=float(args.temporal_thr),
        mask_thr=float(args.mask_thr),
        iou_thr=float(args.iou_thr),
        overlap_thr=float(args.overlap_thr),
        pixel_dist_thr=float(args.pixel_dist_thr),
    )

    n_cells = A.shape[1]
    print(f"ROIs: {n_cells}")
    print(f"Duplicate groups (cos >= {args.cos_thr}): {len(res.groups)}")
    if res.groups:
        for g, rep in zip(res.groups, res.representative_for_group):
            print(f"  group={g}  keep={rep}")
    print(f"Kept: {len(res.kept_indices)}  Dropped: {len(res.dropped_indices)}")

    mapping = {
        "results": str(results),
        "cos_thr": float(args.cos_thr),
        "mask_thr": float(args.mask_thr),
        "iou_thr": float(args.iou_thr),
        "overlap_thr": float(args.overlap_thr),
        "pixel_dist_thr": float(args.pixel_dist_thr),
        "max_dist": float(args.max_dist),
        "temporal_thr": float(args.temporal_thr),
        "n_cells_original": int(n_cells),
        "kept_indices": res.kept_indices,
        "dropped_indices": res.dropped_indices,
        "groups": res.groups,
        "representative_for_group": res.representative_for_group,
    }

    if args.dry_run:
        return

    # Write deduped outputs
    kept = np.array(res.kept_indices, dtype=int)
    A_dedup = A[:, kept]
    save_npz(results / "A_spatial_components_dedup.npz", A_dedup.tocsc())

    if F_dff is not None:
        np.save(results / "F_dff_dedup.npy", F_dff[kept])

    if S_path.exists():
        S = np.load(str(S_path))
        np.save(results / "S_deconv_dedup.npy", S[kept])

    with open(results / "dedup_map.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print("Wrote:")
    print("  A_spatial_components_dedup.npz")
    if F_dff is not None:
        print("  F_dff_dedup.npy")
    if S_path.exists():
        print("  S_deconv_dedup.npy")
    print("  dedup_map.json")


if __name__ == "__main__":
    main()

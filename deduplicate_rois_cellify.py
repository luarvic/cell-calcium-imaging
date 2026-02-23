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
        (or use --temporal_ignore_dist to apply correlation only)
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


def soma_likeness_score(a_norm_2d: np.ndarray, core_mask: np.ndarray | None) -> float:
    """Higher = more soma-like (compact blob)."""
    a_pos = np.maximum(a_norm_2d, 0.0)
    peak = float(a_pos.max()) if a_pos.size else 0.0
    if core_mask is None:
        # fallback: use all positive pixels
        ys, xs = np.where(a_pos > 0)
    else:
        ys, xs = np.where(core_mask)
        if xs.size == 0:
            ys, xs = np.where(a_pos > 0)

    if xs.size == 0:
        return -1e9

    area = float(xs.size)

    # compactness via covariance eigenvalue ratio (penalize elongated shapes)
    y = ys.astype(float)
    x = xs.astype(float)
    y -= y.mean()
    x -= x.mean()
    cov_yy = float(np.mean(y * y)) + 1e-9
    cov_xx = float(np.mean(x * x)) + 1e-9
    cov_yx = float(np.mean(y * x))
    # eigenvalues of 2x2 covariance
    tr = cov_yy + cov_xx
    det = cov_yy * cov_xx - cov_yx * cov_yx
    disc = max(tr * tr - 4.0 * det, 0.0) ** 0.5
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    elong = float(l1 / max(l2, 1e-9))  # >= 1

    # peak density favors compact somas over large diffuse neuropil
    peak_density = peak / max(area, 1.0)
    # final score: favor high peak density and low elongation
    return peak_density / (1.0 + 0.3 * (elong - 1.0))


def pick_cell_representative(
    indices: list[int],
    *,
    A: csc_matrix,
    dims: tuple[int, int] | None,
    mask_thr: float,
    F_dff: np.ndarray | None,
) -> int:
    """Pick one ROI to represent a cell among soma+process ROIs."""
    if dims is None:
        # Fallback: pick strongest activity variance (often soma-dominant)
        if F_dff is not None and F_dff.size > 0:
            vars_ = [float(np.nanvar(F_dff[i])) for i in indices]
            return indices[int(np.argmax(vars_))]
        return indices[0]

    d1, d2 = dims
    # Precompute core masks for candidates
    best = indices[0]
    best_score = -1e18
    for i in indices:
        a = A[:, i].toarray().ravel()
        if a.size != d1 * d2:
            continue
        a2 = a.reshape((d1, d2), order="F")
        a_max = float(np.max(a2))
        core = None
        if a_max > 0:
            core = a2 >= (mask_thr * a_max)
        score = soma_likeness_score(a2, core)
        # tie-breaker: higher trace variance
        if F_dff is not None and F_dff.size > 0:
            score += 1e-3 * float(np.nanvar(F_dff[i]))
        if score > best_score:
            best_score = score
            best = i
    return best


@dataclass
class CellifyResult:
    kept_indices: list[int]
    dropped_indices: list[int]
    groups: list[list[int]]
    representative_for_group: list[int]


def cellify(
    A: csc_matrix,
    *,
    F_dff: np.ndarray | None,
    dims: tuple[int, int] | None,
    max_dist: float,
    mask_thr: float,
    iou_thr: float,
    overlap_thr: float,
    pixel_dist_thr: float,
    soma_nms_dist: float | None = None,
) -> CellifyResult:
    """Collapse CNMF components into *cell-level* ROIs (1 label per soma).

    Goal: produce publication-friendly *one ROI per cell* by:
      1) scoring each ROI for soma-likeness (compact blob)
      2) keeping only soma-like ROIs as representatives
      3) assigning process-like ROIs to the nearest soma (and dropping them)

    This avoids having multiple labels for soma+process fragments.
    """
    n = A.shape[1]
    if n <= 1:
        return CellifyResult(kept_indices=list(range(n)), dropped_indices=[], groups=[], representative_for_group=[])

    if dims is None:
        # Without dims we can't reliably measure morphology; fall back to the old conservative grouping.
        uf = UnionFind(n)
        kept = list(range(n))
        return CellifyResult(kept_indices=kept, dropped_indices=[], groups=[], representative_for_group=[])

    d1, d2 = dims
    cent = centroids_from_A(A, dims)

    # Build core masks once (used for area/overlap/pixel-dist and soma scoring)
    cores, core_sizes = core_masks_from_A(A, mask_thr=mask_thr)
    trees = core_trees_from_indices(cores, dims) if cores is not None else None

    # Soma-likeness score per ROI
    scores = np.full((n,), -1e9, dtype=float)
    for i in range(n):
        if core_sizes[i] == 0:
            continue
        a = A[:, i].toarray().ravel()
        if a.size != d1 * d2:
            continue
        a2 = a.reshape((d1, d2), order="F")
        a_max = float(np.max(a2))
        core = None
        if a_max > 0:
            core = a2 >= (mask_thr * a_max)
        scores[i] = float(soma_likeness_score(a2, core))

    # Heuristics: pick soma candidates by score percentile and a minimal core area.
    # Area threshold removes tiny specks; percentile adapts to each field.
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        finite_scores = np.array([-1e9])
    score_thr = float(np.nanpercentile(finite_scores, 60))  # keep top 40% most soma-like
    area_min = max(15, int(np.nanpercentile(core_sizes.astype(float), 20)))  # adaptive, but at least 15 px
    soma_candidates = [i for i in range(n) if (scores[i] >= score_thr and core_sizes[i] >= area_min)]

    # Fallback: if too strict, at least keep the best-scoring ROI
    if not soma_candidates:
        soma_candidates = [int(np.nanargmax(scores))]

    # Step 1: merge soma candidates that are actually the same soma (very close/overlapping),
    # then pick one representative per soma-cluster.
    uf = UnionFind(n)
    for ii, i in enumerate(soma_candidates):
        for j in soma_candidates[ii + 1:]:
            if np.any(np.isnan(cent[i])) or np.any(np.isnan(cent[j])):
                continue
            dy = float(cent[i, 0] - cent[j, 0])
            dx = float(cent[i, 1] - cent[j, 1])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > float(max_dist):
                continue

            iou, ov = iou_and_overlap(cores[i], cores[j])
            close_pixels = False
            if trees is not None:
                md = min_core_pixel_dist(trees[i], trees[j])
                close_pixels = md <= float(pixel_dist_thr)

            if iou >= float(iou_thr) or ov >= float(overlap_thr) or close_pixels:
                uf.union(i, j)

    comp: dict[int, list[int]] = {}
    for i in soma_candidates:
        r = uf.find(i)
        comp.setdefault(r, []).append(i)

    soma_groups = [sorted(v) for v in comp.values()]
    soma_groups.sort(key=lambda g: g[0])

    soma_reps: list[int] = []
    for g in soma_groups:
        soma_reps.append(pick_cell_representative(g, A=A, dims=dims, mask_thr=mask_thr, F_dff=F_dff))



    # Optional Soma NMS: keep at most one soma representative per neighborhood.
    # This prevents multiple labels on one apparent soma when CNMF over-segments it.
    _nms = float(soma_nms_dist) if soma_nms_dist is not None else float(max_dist) * 1.8
    if len(soma_reps) > 1 and _nms > 0:
        # Sort by soma-likeness score (best first)
        soma_reps_sorted = sorted(soma_reps, key=lambda i: scores[i], reverse=True)
        kept_somas: list[int] = []
        for i in soma_reps_sorted:
            if np.any(np.isnan(cent[i])):
                continue
            yi, xi = cent[i, 0], cent[i, 1]
            too_close = False
            for j in kept_somas:
                yj, xj = cent[j, 0], cent[j, 1]
                dy = float(yi - yj)
                dx = float(xi - xj)
                if (dx * dx + dy * dy) ** 0.5 <= _nms:
                    too_close = True
                    break
            if not too_close:
                kept_somas.append(i)
        soma_reps = kept_somas

    # Step 2: assign every non-soma ROI to the nearest soma rep if it's close enough spatially.
    # (process-like ROIs are dropped; soma reps are kept)
    assignments: dict[int, list[int]] = {rep: [rep] for rep in soma_reps}
    dropped: set[int] = set()

    def _dist(a: int, b: int) -> float:
        dy = float(cent[a, 0] - cent[b, 0])
        dx = float(cent[a, 1] - cent[b, 1])
        return (dx * dx + dy * dy) ** 0.5

    # A bit more permissive than soma merge: processes can extend away from soma center.
    assign_max_dist = float(max_dist) * 1.8

    soma_rep_set = set(soma_reps)
    for i in range(n):
        if i in soma_rep_set:
            continue
        if core_sizes[i] == 0:
            continue

        # Find nearest soma rep by centroid distance
        best_rep = None
        best_d = 1e18
        for rep in soma_reps:
            d = _dist(i, rep)
            if d < best_d:
                best_d = d
                best_rep = rep

        if best_rep is None or best_d > assign_max_dist:
            # Far from any soma rep: keep as separate cell.
            assignments.setdefault(i, [i])
            soma_reps.append(i)
            soma_rep_set.add(i)
            continue

        # Confirm spatial relation by overlap or pixel-distance (prevents accidental merges)
        iou, ov = iou_and_overlap(cores[i], cores[best_rep])
        close_pixels = False
        if trees is not None:
            md = min_core_pixel_dist(trees[i], trees[best_rep])
            close_pixels = md <= float(pixel_dist_thr) * 2.0  # processes can be farther

        if (iou >= float(iou_thr) * 0.5) or (ov >= float(overlap_thr) * 0.5) or close_pixels:
            assignments[best_rep].append(i)
            dropped.add(i)
        else:
            # If this ROI is very close to an existing soma rep, treat it as a process fragment
            # even if overlap is weak (prevents multiple labels on one soma).
            if best_d <= _nms:
                assignments[best_rep].append(i)
                dropped.add(i)
            else:
                # keep as separate cell (safer than over-merging neighbors)
                assignments.setdefault(i, [i])
                soma_reps.append(i)
                soma_rep_set.add(i)

    # Final: groups are the assignment lists with >1 members
    groups = [sorted(v) for v in assignments.values() if len(v) > 1]
    groups.sort(key=lambda g: g[0])

    # Representatives correspond to each multi-member group (first element is rep in our assignment dict)
    rep_for_group = []
    for g in groups:
        # pick rep among the group using soma-likeness (ensures soma kept)
        rep_for_group.append(pick_cell_representative(g, A=A, dims=dims, mask_thr=mask_thr, F_dff=F_dff))

    kept = [i for i in range(n) if i not in dropped]

    # But we want exactly one ROI per cell: keep only the reps of every assignment group
    # (including singleton groups)
    final_kept: list[int] = []
    for rep, members in assignments.items():
        rep2 = pick_cell_representative(members, A=A, dims=dims, mask_thr=mask_thr, F_dff=F_dff)
        final_kept.append(rep2)
    final_kept = sorted(set(final_kept))

    # Anything not in final_kept is dropped (even if not assigned), to ensure 1 ROI per cell.
    dropped_all = sorted([i for i in range(n) if i not in final_kept])

    return CellifyResult(
        kept_indices=final_kept,
        dropped_indices=dropped_all,
        groups=groups,
        representative_for_group=rep_for_group,
    )
def deduplicate(
    A: csc_matrix,
    *,
    cos_thr: float | None,
    F_dff: np.ndarray | None,
    dims: tuple[int, int] | None,
    max_dist: float,
    temporal_thr: float,
    temporal_ignore_dist: bool,
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

            # If still not merged, try temporal correlation (if traces available)
            if not merged and F_dff is not None:
                if temporal_ignore_dist:
                    corr = safe_corr(F_dff[i], F_dff[j])
                    if corr >= float(temporal_thr):
                        uf.union(i, j)
                else:
                    if cent is None:
                        continue
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
        "--temporal_ignore_dist",
        action="store_true",
        help="Ignore spatial distance and use temporal correlation only.",
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
    ap.add_argument("--cellify", action="store_true", help="Collapse soma+process ROIs into one cell-level ROI per cell (publication-friendly).")
    ap.add_argument("--cell_max_dist", type=float, default=15.0, help="Max centroid distance (px) for grouping ROIs into the same cell.")
    ap.add_argument("--cell_iou_thr", type=float, default=0.02, help="Min IoU between core masks to group ROIs into same cell.")
    ap.add_argument("--cell_overlap_thr", type=float, default=0.10, help="Min overlap fraction between core masks to group ROIs into same cell.")
    ap.add_argument("--cell_pixel_dist_thr", type=float, default=8.0, help="Max min pixel distance between core masks to group ROIs into same cell.")
    ap.add_argument("--soma_nms_dist", type=float, default=None, help="Non-maximum suppression distance (px) for soma candidates in --cellify. Keeps only one soma-like ROI per neighborhood. Default: cell_max_dist*1.8.")
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
        temporal_ignore_dist=bool(args.temporal_ignore_dist),
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
        "temporal_ignore_dist": bool(args.temporal_ignore_dist),
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

    
    # Optional: collapse soma+process ROIs into one cell-level ROI per cell
    if args.cellify:
        if dims is None:
            print("[cellify] WARNING: could not determine dims; skipping cellify (needs mmap or dims). Provide --mmap.")
        else:
            A_in = A_dedup
            F_in = (F_dff[kept] if F_dff is not None else None)
            S_in = None
            if S_path.exists():
                S = np.load(str(S_path))
                S_in = S[kept]

            cell_res = cellify(
                A_in,
                F_dff=F_in,
                dims=dims,
                max_dist=float(args.cell_max_dist),
                mask_thr=float(args.mask_thr),
                iou_thr=float(args.cell_iou_thr),
                overlap_thr=float(args.cell_overlap_thr),
                pixel_dist_thr=float(args.cell_pixel_dist_thr),
                soma_nms_dist=float(args.soma_nms_dist) if args.soma_nms_dist is not None else None,
            )
            kept_c = np.array(cell_res.kept_indices, dtype=int)
            A_cells = A_in[:, kept_c]
            save_npz(results / "A_spatial_components_cells.npz", A_cells.tocsc())
            if F_in is not None:
                np.save(results / "F_dff_cells.npy", F_in[kept_c])
            if S_in is not None:
                np.save(results / "S_deconv_cells.npy", S_in[kept_c])

            cell_mapping = {
                "results": str(results),
                "based_on": "dedup",
                "cell_max_dist": float(args.cell_max_dist),
                "cell_iou_thr": float(args.cell_iou_thr),
                "cell_overlap_thr": float(args.cell_overlap_thr),
                "cell_pixel_dist_thr": float(args.cell_pixel_dist_thr),
                "mask_thr": float(args.mask_thr),
                "n_rois_dedup": int(A_in.shape[1]),
                "kept_indices": cell_res.kept_indices,
                "dropped_indices": cell_res.dropped_indices,
                "groups": cell_res.groups,
                "representative_for_group": cell_res.representative_for_group,
            }
            with open(results / "cell_map.json", "w", encoding="utf-8") as f:
                json.dump(cell_mapping, f, indent=2)

            print(f"[cellify] Cells kept: {len(cell_res.kept_indices)}  dropped: {len(cell_res.dropped_indices)}  groups: {len(cell_res.groups)}")


if __name__ == "__main__":
    main()

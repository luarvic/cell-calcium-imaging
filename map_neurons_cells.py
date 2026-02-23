"""
Map neuron index -> ROI location on the image + its trace.

This version DOES NOT rely on cnm_results.hdf5 (since yours isn't in CNMF format).
It uses:
  - results/A_spatial_components.npz (sparse spatial footprints)
  - results/F_dff.npy               (ΔF/F0 traces)
  - a memmap file produced during the pipeline (mc_Corder*.mmap) to get correct dims
    and to build a background image of the same size.

Usage:
  conda activate caiman
  python map_neurons.py --results results --neuron 0 --fr 10

Optional:
  python map_neurons.py --results results --label_all
"""

import argparse
import math
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import load_npz

import caiman as cm


def find_first(patterns, base: Path) -> Path | None:
    for pat in patterns:
        hits = sorted(base.glob(pat))
        if hits:
            return hits[0]
    return None


def load_memmap_dims_and_mean(memmap_path: Path):
    """
    Returns:
      dims: (d1, d2)
      mean_img: (d1, d2) mean across time
    """
    # Yr: (pixels, T), dims: (d1, d2), T: frames
    Yr, dims, T = cm.load_memmap(str(memmap_path))
    mean_vec = np.asarray(Yr.mean(axis=1)).squeeze()  # pixels
    mean_img = mean_vec.reshape(dims, order="F")      # CaImAn footprints use Fortran ordering
    return dims, mean_img


def compute_mask(a: np.ndarray, *, thr_mode: str, thr_frac: float, thr_percentile: float) -> np.ndarray:
    if a.size == 0:
        return np.zeros_like(a, dtype=bool)
    a_max = float(np.max(a))
    if a_max <= 0:
        return np.zeros_like(a, dtype=bool)

    a_norm = a / (a_max + 1e-12)
    if thr_mode == "percentile":
        if not (0.0 <= thr_percentile <= 100.0):
            raise ValueError("thr_percentile must be between 0 and 100.")
        vals = a_norm[a_norm > 0]
        if vals.size == 0:
            return np.zeros_like(a_norm, dtype=bool)
        thr_val = float(np.percentile(vals, thr_percentile))
        return a_norm >= thr_val

    if not (0.0 <= thr_frac <= 1.0):
        raise ValueError("thr_frac must be between 0 and 1.")
    return a_norm >= (thr_frac * float(np.max(a_norm)))


def adjust_label_position(
    x: int,
    y: int,
    placed: list[tuple[int, int]],
    *,
    min_dist: float,
    bounds: tuple[int, int],
) -> tuple[int, int]:
    width, height = bounds

    def in_bounds(px: int, py: int) -> bool:
        return 0 <= px < width and 0 <= py < height

    def ok(px: int, py: int) -> bool:
        if not in_bounds(px, py):
            return False
        for ox, oy in placed:
            dx = float(px - ox)
            dy = float(py - oy)
            if (dx * dx + dy * dy) ** 0.5 < float(min_dist):
                return False
        return True

    if ok(x, y):
        return x, y

    for r in (min_dist, min_dist * 2.0, min_dist * 3.0, min_dist * 4.0):
        for ang in range(0, 360, 45):
            dx = int(round(r * math.cos(math.radians(ang))))
            dy = int(round(r * math.sin(math.radians(ang))))
            nx, ny = x + dx, y + dy
            if ok(nx, ny):
                return nx, ny

    return x, y


def plot_one(
    neuron_idx: int,
    A,
    F_dff,
    bg_img,
    dims,
    outpath: Path | None,
    *,
    thr_mode: str,
    thr_frac: float,
    thr_percentile: float,
):
    d1, d2 = dims

    # Spatial footprint for neuron
    a = A[:, neuron_idx].toarray().reshape((d1, d2), order="F")
    if np.max(a) > 0:
        a_norm = a / (np.max(a) + 1e-12)
    else:
        a_norm = a

    mask = compute_mask(a_norm, thr_mode=thr_mode, thr_frac=thr_frac, thr_percentile=thr_percentile)

    fig = plt.figure(figsize=(14, 5))

    # ROI overlay
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(bg_img, aspect="auto")
    if mask.any():
        ax1.contour(mask, levels=[0.5], linewidths=1)
    ax1.set_title(f"Neuron {neuron_idx}: ROI overlay")
    ax1.axis("off")

    # Trace
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(F_dff[neuron_idx], linewidth=1.0)
    ax2.set_title(f"Neuron {neuron_idx}: ΔF/F₀")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("ΔF/F₀")

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def label_all_rois(
    A,
    bg_img,
    dims,
    outpath: Path,
    *,
    thr_mode: str,
    thr_frac: float,
    thr_percentile: float,
    label_mode: str,
    jitter_labels: bool,
    min_label_dist: float,
):
    d1, d2 = dims
    n_cells = A.shape[1]

    plt.figure(figsize=(12, 10))
    plt.imshow(bg_img, aspect="auto")
    plt.axis("off")
    plt.title("ROI indices (match row index in F_dff)")

    placed: list[tuple[int, int]] = []
    for i in range(n_cells):
        a = A[:, i].toarray().reshape((d1, d2), order="F")
        a_max = float(a.max())
        if a_max <= 0:
            continue

        a_norm = a / (a_max + 1e-12)
        mask = compute_mask(a_norm, thr_mode=thr_mode, thr_frac=thr_frac, thr_percentile=thr_percentile)

        if label_mode == "centroid":
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            x, y = int(xs.mean()), int(ys.mean())
        else:
            flat_idx = int(np.argmax(a_norm))
            y, x = np.unravel_index(flat_idx, (d1, d2), order="C")

        if jitter_labels:
            x, y = adjust_label_position(x, y, placed, min_dist=min_label_dist, bounds=(d2, d1))
        placed.append((x, y))

        plt.text(
            x,
            y,
            str(i),
            fontsize=7,
            color="white",
            ha="center",
            va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=0.2),
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="Results folder with A_spatial_components.npz and F_dff.npy")
    ap.add_argument("--neuron", type=int, default=0, help="Neuron index to visualize")
    ap.add_argument("--fr", type=float, required=False, help="Frame rate (Hz). Only used if you later add time axis.")
    ap.add_argument("--thr_frac", type=float, default=0.2, help="ROI threshold fraction for contour/mask (0..1)")
    ap.add_argument(
        "--thr_mode",
        choices=["frac", "percentile"],
        default="percentile",
        help="Mask threshold mode: fraction of max or percentile of nonzero footprint values.",
    )
    ap.add_argument(
        "--thr_percentile",
        type=float,
        default=90.0,
        help="Percentile (0..100) for per-ROI thresholding when --thr_mode percentile.",
    )
    ap.add_argument("--label_all", action="store_true", help="Create a labeled ROI map for all neurons")
    ap.add_argument(
        "--label_mode",
        choices=["peak", "centroid"],
        default="peak",
        help="Label placement: peak footprint pixel or centroid of thresholded mask.",
    )
    ap.add_argument(
        "--jitter_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offset labels that are too close to avoid overlaps.",
    )
    ap.add_argument(
        "--min_label_dist",
        type=float,
        default=6.0,
        help="Minimum pixel distance between label positions when jittering.",
    )
    ap.add_argument("--dedup", action="store_true", help="Use *_dedup outputs if present (from deduplicate_rois.py)")
    ap.add_argument("--cells", action="store_true", help="Use *_cells outputs if present (cell-level ROIs from deduplicate_rois_cellify.py --cellify).")
    ap.add_argument("--mmap", default=None, help="Path to .mmap file (optional). If omitted, auto-detect in results folder.")
    ap.add_argument("--out", default=None, help="Output PNG for single neuron plot (optional)")
    args = ap.parse_args()

    results = Path(args.results).expanduser().resolve()
    if args.cells:
        A_path = results / "A_spatial_components_cells.npz"
        F_path = results / "F_dff_cells.npy"
    elif args.dedup:
        A_path = results / "A_spatial_components_dedup.npz"
        F_path = results / "F_dff_dedup.npy"
    else:
        A_path = results / "A_spatial_components.npz"
        F_path = results / "F_dff.npy"

    if not A_path.exists():
        raise SystemExit(f"Missing: {A_path}")
    if not F_path.exists():
        raise SystemExit(f"Missing: {F_path}")

    # Pick memmap: user-provided --mmap OR auto-detect inside the results folder
    if args.mmap:
        memmap = Path(args.mmap).expanduser().resolve()
    else:
        # Prefer the C-order memmap we created in the pipeline
        candidates = sorted(results.glob("mc_Corder*.mmap"))
        if not candidates:
            candidates = sorted(results.glob("*.mmap"))
        memmap = candidates[0] if candidates else None

    if memmap is None or not memmap.exists():
        raise SystemExit(
            "Could not find a .mmap file. "
            "Expected mc_Corder*.mmap in the results folder. "
            "Re-run the pipeline or pass --mmap explicitly."
        )

    print("Using memmap:", memmap)

    # Load outputs
    A = load_npz(str(A_path))              # (pixels, n_cells)
    F_dff = np.load(str(F_path))           # (n_cells, T)

    # Build background image with matching dims
    dims, bg_img = load_memmap_dims_and_mean(memmap)

    pixels, n_cells = A.shape
    if pixels != dims[0] * dims[1]:
        raise SystemExit(
            f"Mismatch: A has {pixels} pixels but memmap dims {dims} imply {dims[0]*dims[1]} pixels.\n"
            f"This means you're mixing outputs from different runs/files."
        )

    if args.neuron < 0 or args.neuron >= n_cells:
        raise SystemExit(f"--neuron out of range. Must be 0..{n_cells-1}")

    # Single neuron plot
    outpath = Path(args.out).expanduser().resolve() if args.out else None
    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
    plot_one(
        args.neuron,
        A,
        F_dff,
        bg_img,
        dims,
        outpath,
        thr_mode=args.thr_mode,
        thr_frac=args.thr_frac,
        thr_percentile=args.thr_percentile,
    )

    # Labeled map for all ROIs
    if args.label_all:
        out_all = results / ("roi_index_map_cells.png" if args.cells else ("roi_index_map_dedup.png" if args.dedup else "roi_index_map.png"))
        label_all_rois(
            A,
            bg_img,
            dims,
            out_all,
            thr_mode=args.thr_mode,
            thr_frac=args.thr_frac,
            thr_percentile=args.thr_percentile,
            label_mode=args.label_mode,
            jitter_labels=args.jitter_labels,
            min_label_dist=args.min_label_dist,
        )
        print("Saved:", out_all)


if __name__ == "__main__":
    main()

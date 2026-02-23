"""
Plot CaImAn results: ΔF/F0 traces for ALL neurons from results/F_dff.npy (or S_deconv.npy)

Usage examples:

  # PNG pages + (optional) heatmap
  conda activate caiman
  python plot_traces.py --results results/well1-000 --fr 10 --zscore --heatmap

  # PNG pages + ONE multi-page PDF (recommended "joint plot" per TIFF)
  python plot_traces.py --results results/well1-000 --fr 10 --zscore --pdf

Outputs (by default in <results>/plots):
  - traces_grid_page_001.png, traces_grid_page_002.png, ...
  - mean_trace.png
  - all_traces_heatmap.png (if --heatmap)
  - traces_all_neurons.pdf (if --pdf)

Notes:
- Use --use F (default) to plot ΔF/F0 (F_dff.npy)
- Use --use S to plot deconvolved activity (S_deconv.npy)
- --zscore applies robust per-neuron scaling for readability in grids/heatmap
"""

import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_array(results_dir: Path, which: str, dedup: bool, cells: bool) -> np.ndarray:
    which_l = which.lower().strip()
    if which_l in ("f", "fdff", "f_dff", "dff"):
        if cells:
            fname = results_dir / "F_dff_cells.npy"
        else:
            fname = results_dir / ("F_dff_dedup.npy" if dedup else "F_dff.npy")
    elif which_l in ("s", "deconv", "s_deconv", "spikes"):
        if cells:
            fname = results_dir / "S_deconv_cells.npy"
        else:
            fname = results_dir / ("S_deconv_dedup.npy" if dedup else "S_deconv.npy")
    else:
        raise ValueError("Unknown --use value. Use 'F' (default) or 'S'.")

    if not fname.exists():
        if cells:
            raise FileNotFoundError(
                f"Missing file: {fname}\n"
                "You passed --cells but cell-level outputs were not found. "
                "Run deduplicate_rois_cellify.py with --cellify first (or omit --cells)."
            )
        if dedup:
            raise FileNotFoundError(
                f"Missing file: {fname}\n"
                "You passed --dedup but deduplicated outputs were not found. "
                "Run deduplicate_rois.py first (or omit --dedup)."
            )
        raise FileNotFoundError(f"Missing file: {fname}")

    # Load normally (these are not gigantic compared to raw movies)
    return np.load(fname, mmap_mode=None)


def robust_zscore_rows(X: np.ndarray) -> np.ndarray:
    """
    Robust row-wise scaling to make many traces comparable:
    z = (x - median) / (1.4826*MAD)
    """
    med = np.nanmedian(X, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(X - med), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1.0, mad)
    return (X - med) / (1.4826 * mad)


def save_heatmap(X: np.ndarray, t: np.ndarray, out_path: Path, title: str, cbar_label: str) -> None:
    fig = plt.figure(figsize=(14, 7))
    plt.imshow(
        X,
        aspect="auto",
        interpolation="nearest",
        extent=[float(t[0]), float(t[-1]), 0, X.shape[0]],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")
    plt.title(title)
    plt.colorbar(label=cbar_label)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_grid_page_figure(
    block: np.ndarray,
    t: np.ndarray,
    start_idx: int,
    ncols: int,
    linewidth: float,
    y_label: str,
    suptitle: str,
) -> plt.Figure:
    """
    Create a single figure containing a grid of traces in 'block'.
    block shape: (n_traces_on_page, T)
    """
    n_traces = block.shape[0]
    nrows = math.ceil(n_traces / ncols)

    fig_w = 14
    fig_h = max(6, 2.0 * nrows)  # scale with rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    for i in range(n_traces):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.plot(t, block[i], linewidth=linewidth)
        ax.set_title(f"Neuron {start_idx + i}", fontsize=9)
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.tick_params(axis="both", labelsize=8)

        # reduce clutter
        if i % ncols != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(y_label, fontsize=8)

        # only bottom row gets x labels
        if i < (nrows - 1) * ncols:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="Results directory containing F_dff.npy / S_deconv.npy")
    ap.add_argument("--fr", type=float, required=True, help="Frame rate (Hz) used for time axis.")
    ap.add_argument("--use", default="F", help="Which array to plot: F (ΔF/F0) or S (deconvolved). Default F.")
    ap.add_argument("--dedup", action="store_true", help="Use *_dedup outputs if present (from deduplicate_rois.py)")
    ap.add_argument("--cells", action="store_true", help="Use *_cells outputs if present (from deduplicate_rois_cellify.py --cellify)")
    ap.add_argument("--plots_dir", default="plots", help="Subfolder inside results to write plots into.")
    ap.add_argument("--traces_per_page", type=int, default=64, help="How many traces per grid page image.")
    ap.add_argument("--ncols", type=int, default=4, help="Number of columns in trace grid.")
    ap.add_argument("--zscore", action="store_true",
                    help="Robust z-score each neuron before grid plotting (helps compare many traces).")
    ap.add_argument("--heatmap", action="store_true",
                    help="Also save a heatmap overview (neurons x time).")
    ap.add_argument("--pdf", action="store_true",
                    help="Also write one multi-page PDF with all trace grid pages.")
    ap.add_argument("--linewidth", type=float, default=0.8, help="Line width for traces.")
    args = ap.parse_args()

    results_dir = Path(args.results).expanduser().resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    if args.cells and args.dedup:
        raise SystemExit("Use only one of --cells or --dedup.")

    X = load_array(results_dir, args.use, dedup=bool(args.dedup), cells=bool(args.cells))
    if X.ndim != 2:
        raise SystemExit(f"Expected 2D array (neurons, frames). Got shape={X.shape}")

    n_neurons, T = X.shape
    fr = float(args.fr)
    t = np.arange(T, dtype=float) / fr

    plots_dir = results_dir / args.plots_dir
    ensure_dir(plots_dir)

    title_kind = "ΔF/F₀" if args.use.lower().startswith("f") else "Deconvolved (S)"
    y_label = "ΔF/F₀" if args.use.lower().startswith("f") else "S"
    cbar_label = "z-scored" if args.zscore else "signal"

    # Scale for readability on grids/heatmap if requested
    X_plot = robust_zscore_rows(X) if args.zscore else X

    # Optional heatmap overview
    if args.heatmap:
        save_heatmap(
            X_plot,
            t,
            plots_dir / "all_traces_heatmap.png",
            title=f"All neurons heatmap ({title_kind})",
            cbar_label=cbar_label,
        )

    # Optional multi-page PDF
    pdf = PdfPages(plots_dir / "traces_all_neurons.pdf") if args.pdf else None

    traces_per_page = int(args.traces_per_page)
    ncols = int(args.ncols)
    n_pages = math.ceil(n_neurons / traces_per_page)

    for page in range(n_pages):
        start = page * traces_per_page
        end = min(start + traces_per_page, n_neurons)
        block = X_plot[start:end]

        fig = make_grid_page_figure(
            block=block,
            t=t,
            start_idx=start,
            ncols=ncols,
            linewidth=float(args.linewidth),
            y_label=y_label,
            suptitle=f"Traces ({title_kind}) — neurons {start}–{end-1}",
        )

        out_png = plots_dir / f"traces_grid_page_{page+1:03d}.png"
        fig.savefig(out_png, dpi=200)

        if pdf is not None:
            pdf.savefig(fig)

        plt.close(fig)

    if pdf is not None:
        pdf.close()

    # Mean trace (on original scale, not z-scored)
    mean_trace = np.nanmean(X, axis=0)
    fig = plt.figure(figsize=(14, 4))
    plt.plot(t, mean_trace, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel(f"Mean {y_label}")
    plt.title(f"Mean across neurons ({title_kind})")
    plt.tight_layout()
    fig.savefig(plots_dir / "mean_trace.png", dpi=200)
    plt.close(fig)

    print(f"Loaded {n_neurons} neurons, {T} frames.")
    print(f"Saved plots to: {plots_dir}")
    if args.pdf:
        print(f"Saved PDF: {plots_dir / 'traces_all_neurons.pdf'}")
    if args.heatmap:
        print(f"Saved heatmap: {plots_dir / 'all_traces_heatmap.png'}")


if __name__ == "__main__":
    main()

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", default="results", help="Root results folder created by caiman_pipeline.py")
    ap.add_argument(
        "--results",
        nargs="+",
        default=None,
        help="Optional: one or more specific results folders to postprocess (e.g. results/well1-000).",
    )
    ap.add_argument("--fr", type=float, required=True, help="Frame rate (Hz)")
    ap.add_argument("--plots_dir", default="plots", help="Subfolder inside each results folder to write plots into")
    ap.add_argument("--thr_frac", type=float, default=0.2, help="ROI threshold fraction for map_neurons.py labeling")
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--dedup", action="store_true", help="Run deduplicate_rois.py before plotting/mapping")
    ap.add_argument("--cellify", action="store_true", help="After dedup, collapse soma+process ROIs into one cell-level ROI per cell (writes *_cells files).")
    ap.add_argument("--cell_max_dist", type=float, default=15.0)
    ap.add_argument("--cell_iou_thr", type=float, default=0.02)
    ap.add_argument("--cell_overlap_thr", type=float, default=0.10)
    ap.add_argument("--cell_pixel_dist_thr", type=float, default=8.0)
    ap.add_argument("--soma_nms_dist", type=float, default=None, help="Non-maximum suppression distance for soma candidates (pixels)")
    ap.add_argument(
        "--dedup_temporal_thr",
        type=float,
        default=0.8,
        help="Min correlation between traces for deduplication.",
    )
    ap.add_argument(
        "--dedup_cos_thr",
        type=float,
        default=0.0,
        help="Cosine similarity threshold for spatial deduplication (0 disables).",
    )
    ap.add_argument(
        "--dedup_ignore_dist",
        action="store_true",
        help="Ignore spatial distance and use temporal correlation only.",
    )
    args = ap.parse_args()

    if args.cellify and not args.dedup:
        raise SystemExit("--cellify requires --dedup (cellify runs on deduplicated ROIs).")

    if args.results:
        folders = [Path(p).expanduser().resolve() for p in args.results]
    else:
        outroot = Path(args.outroot).expanduser().resolve()
        folders = sorted([p for p in outroot.iterdir() if p.is_dir() and (p / "F_dff.npy").exists()])

    # Validate
    folders = [p for p in folders if p.is_dir()]
    folders = [p for p in folders if (p / "F_dff.npy").exists()]
    if not folders:
        if args.results:
            raise SystemExit("No valid results folders provided (must contain F_dff.npy).")
        raise SystemExit(f"No result folders found under: {outroot}")

    for d in folders:
        print("\n=== Postprocess:", d.name, "===")

        # 0) optional dedup
        if args.dedup:
            cmd0 = [
                sys.executable,
                "deduplicate_rois_cellify.py",
                "--results",
                str(d),
                "--cos_thr",
                str(args.dedup_cos_thr),
                "--temporal_thr",
                str(args.dedup_temporal_thr),
            ]
            if args.dedup_ignore_dist:
                cmd0.append("--temporal_ignore_dist")
            if args.cellify:
                cmd0.append("--cellify")
                cmd0 += ["--cell_max_dist", str(args.cell_max_dist),
                         "--cell_iou_thr", str(args.cell_iou_thr),
                         "--cell_overlap_thr", str(args.cell_overlap_thr),
                         "--cell_pixel_dist_thr", str(args.cell_pixel_dist_thr)]
            if args.soma_nms_dist is not None:
                cmd0 += ["--soma_nms_dist", str(args.soma_nms_dist)]
            subprocess.check_call(cmd0)

        # 1) traces (all neurons)
        cmd = [sys.executable, "plot_traces.py", "--results", str(d), "--fr", str(args.fr), "--plots_dir", str(args.plots_dir)]
        if args.cellify:
            cmd.append("--cells")
        elif args.dedup:
            cmd.append("--dedup")
        if args.zscore: cmd.append("--zscore")
        if args.heatmap: cmd.append("--heatmap")
        if args.pdf: cmd.append("--pdf")
        subprocess.check_call(cmd)

        # 2) labeled ROI map
        cmd2 = [sys.executable, "map_neurons_cells.py", "--results", str(d), "--label_all", "--thr_frac", str(args.thr_frac)]
        if args.cellify:
            cmd2.append("--cells")
        elif args.dedup:
            cmd2.append("--dedup")
        subprocess.check_call(cmd2)

    print("\nAll postprocessing done.")

if __name__ == "__main__":
    main()

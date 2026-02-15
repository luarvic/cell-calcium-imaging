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
    args = ap.parse_args()

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

        # 1) traces (all neurons)
        cmd = [sys.executable, "plot_traces.py", "--results", str(d), "--fr", str(args.fr), "--plots_dir", str(args.plots_dir)]
        if args.zscore: cmd.append("--zscore")
        if args.heatmap: cmd.append("--heatmap")
        if args.pdf: cmd.append("--pdf")
        subprocess.check_call(cmd)

        # 2) labeled ROI map
        cmd2 = [sys.executable, "map_neurons.py", "--results", str(d), "--label_all", "--thr_frac", str(args.thr_frac)]
        subprocess.check_call(cmd2)

    print("\nAll postprocessing done.")

if __name__ == "__main__":
    main()

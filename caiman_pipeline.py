import argparse
import os
from pathlib import Path
from shutil import copy2
import numpy as np

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from scipy.sparse import save_npz


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def expand_inputs(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        item_p = Path(item).expanduser()
        # glob support
        if any(ch in item for ch in ["*", "?", "["]):
            files.extend([p.resolve() for p in item_p.parent.glob(item_p.name) if p.is_file()])
        else:
            if item_p.is_dir():
                files.extend(sorted([p.resolve() for p in item_p.glob("*.tif") if p.is_file()]))
                files.extend(sorted([p.resolve() for p in item_p.glob("*.tiff") if p.is_file()]))
            else:
                files.append(item_p.resolve())
    # de-dup while preserving order
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def event_times_from_S(S: np.ndarray, fr: float, z: float = 3.0) -> list[np.ndarray]:
    events = []
    for i in range(S.shape[0]):
        s = np.ravel(S[i]).astype(float)
        if s.size == 0 or np.all(s == 0) or np.std(s) == 0:
            events.append(np.array([], dtype=float))
            continue
        thr = z * np.std(s)
        above = np.where(s > thr)[0]
        if above.size == 0:
            events.append(np.array([], dtype=float))
            continue
        splits = np.where(np.diff(above) > 1)[0] + 1
        groups = np.split(above, splits)
        ev_frames = np.array([g[0] for g in groups if g.size > 0], dtype=int)
        events.append(ev_frames / fr)
    return events


def process_one_tiff(fn: Path, outdir: Path, fr: float, decay_time: float, gSig: int,
                     K: int, pw_rigid: bool, max_shifts: int, strides: int, overlaps: int,
                     event_z: float, n_processes: int, merge_thr: float | None) -> None:
    ensure_dir(outdir)

    print(f"\n=== Processing: {fn.name} ===")
    print(f"Output: {outdir}")

    print("[1/7] Loading TIFF")
    m = cm.load(str(fn))
    print(f"      shape={m.shape}, dtype={m.dtype}")
    if len(m.shape) != 3:
        raise SystemExit(f"Expected 3D movie (T,d1,d2). Got {m.shape}")
    T, d1, d2 = m.shape
    if T < 30:
        raise SystemExit(f"Too few frames: {T}")

    opts_dict = {
        "data": {"fnames": [str(fn)], "fr": fr, "decay_time": decay_time},
        "motion": {"pw_rigid": pw_rigid, "max_shifts": (max_shifts, max_shifts)},
        "init": {"method_init": "greedy_roi", "gSig": (gSig, gSig), "K": K},
        "temporal": {"p": 1},
    }
    if merge_thr is not None:
        opts_dict["merging"] = {"merge_thr": float(merge_thr)}
    if pw_rigid:
        opts_dict["motion"].update({
            "strides": (strides, strides),
            "overlaps": (overlaps, overlaps),
        })

    opts = params.CNMFParams(params_dict=opts_dict)

    print("[2/7] Motion correction")
    mc = MotionCorrect([str(fn)], **opts.get_group("motion"))
    mc.motion_correct(save_movie=True)

    # mc.mmap_file may be list
    fname_mc = mc.mmap_file
    if isinstance(fname_mc, (list, tuple)):
        fname_mc = fname_mc[0]
    if not fname_mc or not os.path.exists(fname_mc):
        raise SystemExit(f"Motion correction memmap missing: {mc.mmap_file}")

    print("[3/7] Convert to C-order memmap (fixes F-order CNMF error)")
    fname_c = cm.save_memmap([fname_mc], base_name=str(outdir / "mc_Corder"), order="C")
    if not os.path.exists(fname_c):
        raise SystemExit(f"Failed to create C-order memmap: {fname_c}")

    # Some CaImAn builds still write memmap elsewhere; enforce copy into outdir
    fname_c_path = Path(fname_c).resolve()
    target = outdir / fname_c_path.name
    if fname_c_path.parent != outdir:
        copy2(fname_c_path, target)
        fname_c = str(target)
        print("      Copied C-order memmap to:", fname_c)

    print("[4/7] CNMF fit")
    opts.change_params({"data": {"fnames": [fname_c]}})
    cnm = cnmf.CNMF(n_processes=n_processes, params=opts)
    cnm.fit_file(motion_correct=False)

    print("[5/7] ΔF/F0 extraction")
    cnm.estimates.detrend_df_f(quantileMin=8, frames_window=int(fr * 30))
    F_dff = cnm.estimates.F_dff
    print("      detected ROIs:", F_dff.shape[0])

    print("[6/7] Deconvolution + event times")
    if cnm.estimates.S is None:
        cnm.estimates.deconvolve(cnm.params)
    S = cnm.estimates.S
    if S is None:
        raise SystemExit("Deconvolution failed: cnm.estimates.S is None")

    events_sec = event_times_from_S(S, fr=fr, z=event_z)

    print("[7/7] Saving outputs")
    np.save(outdir / "F_dff.npy", F_dff)
    np.save(outdir / "S_deconv.npy", S)
    np.save(outdir / "events_sec.npy", np.array(events_sec, dtype=object))

    # Save spatial footprints A (pixels x n_cells) — needed for mapping ROIs to image
    A = cnm.estimates.A
    save_npz(outdir / "A_spatial_components.npz", A.tocsc())

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"input={fn}\n")
        f.write(f"shape={m.shape}\n")
        f.write(f"fr={fr}\n")
        f.write(f"decay_time={decay_time}\n")
        f.write(f"gSig={gSig}\n")
        f.write(f"K={K}\n")
        f.write(f"pw_rigid={pw_rigid}\n")
        f.write(f"merge_thr={merge_thr}\n")
        f.write(f"mmap_c={fname_c}\n")
        f.write(f"n_cells={F_dff.shape[0]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", "-i", nargs="+", required=True,
                    help="TIFFs, a directory, or globs. Examples: well*.tif OR ./tiffs/ OR ./tiffs/*.tif")
    ap.add_argument("--outroot", default="results", help="Root output folder. Each TIFF writes to outroot/<basename>/")
    ap.add_argument("--fr", type=float, required=True, help="Frame rate (Hz)")
    ap.add_argument("--decay_time", type=float, default=0.4)
    ap.add_argument("--gSig", type=int, default=5)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--pw_rigid", action="store_true")
    ap.add_argument("--max_shifts", type=int, default=8)
    ap.add_argument("--strides", type=int, default=64)
    ap.add_argument("--overlaps", type=int, default=32)
    ap.add_argument("--event_z", type=float, default=3.0)
    ap.add_argument("--n_processes", type=int, default=1)
    ap.add_argument(
        "--merge_thr",
        type=float,
        default=None,
        help="CNMF merging threshold. If set, passed as params['merging']['merge_thr'] (e.g. 0.85).",
    )
    args = ap.parse_args()

    inputs = expand_inputs(args.inputs)
    if not inputs:
        raise SystemExit("No TIFF files found from --inputs")

    outroot = Path(args.outroot).expanduser().resolve()
    ensure_dir(outroot)

    for fn in inputs:
        if not fn.exists():
            print("Skipping (missing):", fn)
            continue
        outdir = outroot / fn.stem
        process_one_tiff(
            fn=fn,
            outdir=outdir,
            fr=args.fr,
            decay_time=args.decay_time,
            gSig=args.gSig,
            K=args.K,
            pw_rigid=args.pw_rigid,
            max_shifts=args.max_shifts,
            strides=args.strides,
            overlaps=args.overlaps,
            event_z=args.event_z,
            n_processes=args.n_processes,
            merge_thr=args.merge_thr,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()

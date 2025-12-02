import argparse
from pathlib import Path

import numpy as np

from src.metrics.uiqi import uiqi
from src.metrics.ssim_wrap import ssim01
from src.metrics.fsim import fsim
from src.io.dicom_png import (
    is_dicom,
    read_dicom_hu,
    read_gray01,
    window_hu,
    window_img01,
)


def load_ref01(path: Path, wl: float, ww: float) -> np.ndarray:
    """Load reference CT slice and map to [0,1]."""
    if is_dicom(path):
        hu = read_dicom_hu(path)
        x = window_hu(hu, wl, ww)
    else:
        g = read_gray01(path)
        x = window_img01(g)
    return np.clip(x.astype(np.float32), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/real", help="clean reference images (PNG/DICOM)")
    ap.add_argument("--out", default="data/outputs", help="folder with enhanced outputs (.npy)")
    ap.add_argument(
        "--mode",
        default="soft",
        choices=["soft", "lung"],
        help="CT window preset for reference",
    )
    args = ap.parse_args()

    if args.mode == "lung":
        wl, ww = -600, 1500
    else:
        wl, ww = 40, 400

    p_ref = Path(args.ref)
    p_out = Path(args.out)

    rows = []
    print("stem, UIQI_clahe, SSIM_clahe, FSIM_clahe, UIQI_ngc, SSIM_ngc, FSIM_ngc, UIQI_prop, SSIM_prop, FSIM_prop")

    for r in sorted(p_ref.iterdir()):
        if r.is_dir():
            continue
        stem = r.stem

        ref01 = load_ref01(r, wl, ww)

        cla_path  = p_out / f"{stem}_clahe.npy"
        ngc_path  = p_out / f"{stem}_ngcclahe.npy"
        prop_path = p_out / f"{stem}_proposed.npy"

        if not (cla_path.exists() and ngc_path.exists() and prop_path.exists()):
            print(f"# skip {stem}: some outputs missing")
            continue

        cla  = np.load(cla_path).astype(np.float32)
        ngc  = np.load(ngc_path).astype(np.float32)
        prop = np.load(prop_path).astype(np.float32)

        # ensure all are in [0,1]
        for x in (cla, ngc, prop):
            x[x < 0.0] = 0.0
            x[x > 1.0] = 1.0

        u_cla  = uiqi(ref01, cla)
        s_cla  = ssim01(ref01, cla)
        f_cla  = fsim(ref01, cla)

        u_ngc  = uiqi(ref01, ngc)
        s_ngc  = ssim01(ref01, ngc)
        f_ngc  = fsim(ref01, ngc)

        u_prop = uiqi(ref01, prop)
        s_prop = ssim01(ref01, prop)
        f_prop = fsim(ref01, prop)

        rows.append(
            (u_cla, s_cla, f_cla, u_ngc, s_ngc, f_ngc, u_prop, s_prop, f_prop)
        )

        print(
            f"{stem},{u_cla:.4f},{s_cla:.4f},{f_cla:.4f},"
            f"{u_ngc:.4f},{s_ngc:.4f},{f_ngc:.4f},"
            f"{u_prop:.4f},{s_prop:.4f},{f_prop:.4f}"
        )

    if not rows:
        print("\nNo rows collected – check that ref/out folders and filenames match.")
        return

    rows = np.array(rows)
    mean_vals = rows.mean(axis=0)

    print("\nMeans over images:")
    print("CLAHE   UIQI/SSIM/FSIM = "
          f"{mean_vals[0]:.4f} {mean_vals[1]:.4f} {mean_vals[2]:.4f}")
    print("NGC     UIQI/SSIM/FSIM = "
          f"{mean_vals[3]:.4f} {mean_vals[4]:.4f} {mean_vals[5]:.4f}")
    print("Proposed UIQI/SSIM/FSIM = "
          f"{mean_vals[6]:.4f} {mean_vals[7]:.4f} {mean_vals[8]:.4f}")


if __name__ == "__main__":
    main()

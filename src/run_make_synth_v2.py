import argparse
from pathlib import Path

import numpy as np

from src.io.dicom_png import (
    is_dicom,
    read_dicom_hu,
    read_gray01,
    window_hu,
    window_img01,
)
from src.utils.degrade import degrade_low_contrast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="data/real",
        help="folder of clean images (PNG/JPG/DICOM)",
    )
    ap.add_argument(
        "--dst",
        default="data/synth",
        help="folder to save degraded images (.npy, float32 in [0,1])",
    )
    ap.add_argument(
        "--mode",
        default="soft",
        choices=["soft", "lung"],
        help="CT window preset (soft tissue vs lung)",
    )
    ap.add_argument(
        "--strength",
        default="medium",
        choices=["mild", "medium", "strong"],
        help="amount of synthetic contrast reduction",
    )
    args = ap.parse_args()

    # CT window presets similar to those used in the base paper
    if args.mode == "lung":
        wl, ww = -600, 1500
    else:  # "soft" – brain / soft-tissue style window
        wl, ww = 40, 400

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for p in sorted(src.iterdir()):
        if p.is_dir():
            continue

        # --- load and window to [0,1] ---
        if is_dicom(p):
            hu = read_dicom_hu(p)
            img01 = window_hu(hu, wl, ww)      # float in [0,1]
        else:
            g = read_gray01(p)
            img01 = window_img01(g)            # float in [0,1]

        # --- apply low-contrast degradation (no noise) ---
        deg01 = degrade_low_contrast(img01, strength=args.strength)

        # save as lossless .npy (float32 in [0,1])
        out_path = dst / f"{p.stem}.npy"
        np.save(out_path, deg01.astype(np.float32))
        print(f"saved {out_path}")

    print(f"\nSynthetic degraded set saved to: {dst}")


if __name__ == "__main__":
    main()

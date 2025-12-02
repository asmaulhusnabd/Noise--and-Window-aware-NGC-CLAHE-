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
from src.enhan.clahe_baseline import clahe_baseline
from src.enhan.ngc_clahe import ngc_clahe
from src.enhan.nw_gc_clahe import nw_gc_clahe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="data/synth",
        help="folder of degraded images (.npy or PNG/DICOM)",
    )
    ap.add_argument(
        "--out",
        default="data/outputs",
        help="where to write enhanced images (.npy)",
    )
    ap.add_argument(
        "--mode",
        default="soft",
        choices=["soft", "lung"],
        help="window preset if loading PNG/DICOM directly",
    )
    args = ap.parse_args()

    # CT window presets (only used if src has DICOM/PNG instead of .npy)
    if args.mode == "lung":
        wl, ww = -600, 1500
    else:  # soft-tissue / brain style
        wl, ww = 40, 400

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for p in sorted(src.iterdir()):
        if p.is_dir():
            continue

        stem = p.stem

        # --- load degraded image as float32 in [0,1] ---
        if p.suffix.lower() == ".npy":
            img01 = np.load(p).astype(np.float32)
            # in case someone saved as 0-255 by mistake
            if img01.max() > 1.001:
                img01 = img01 / 255.0
            img01 = np.clip(img01, 0.0, 1.0)
        else:
            # fallback: load DICOM / PNG and window on the fly
            if is_dicom(p):
                hu = read_dicom_hu(p)
                img01 = window_hu(hu, wl, ww)
            else:
                g = read_gray01(p)
                img01 = window_img01(g)

        # --- method 1: plain CLAHE baseline ---
        cla = clahe_baseline(img01, clip=2.0, tile=(8, 8))

        # --- method 2: NGC-CLAHE (paper baseline) ---
        base = ngc_clahe(
            img01,
            gamma=0.95,
            clip=2.0,
            tile=(8, 8),
        )

        # --- method 3: Proposed NW-GC-CLAHE ---
        prop, _maps = nw_gc_clahe(
            img01,
            gamma=0.95,
            clip_cons=1.0,
            clip_agg=3.0,
            tile=(8, 8),
        )

        np.save(out / f"{stem}_clahe.npy", cla.astype(np.float32))
        np.save(out / f"{stem}_ngcclahe.npy", base.astype(np.float32))
        np.save(out / f"{stem}_proposed.npy", prop.astype(np.float32))

        print(f"processed {p.name}")

    print(f"\nEnhanced outputs saved to {out}")


if __name__ == "__main__":
    main()

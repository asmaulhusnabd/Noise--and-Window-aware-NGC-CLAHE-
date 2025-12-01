import argparse, numpy as np
from pathlib import Path
from src.io.dicom_png import is_dicom, read_dicom_hu, read_gray01, window_hu, window_img01
from src.utils.degrade import degrade_low_contrast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/real", help="folder of clean images (PNG or DICOM)")
    ap.add_argument("--dst", default="data/synth", help="where to write degraded images (npy)")
    ap.add_argument("--mode", default="soft", choices=["soft","lung"], help="window preset")
    args = ap.parse_args()

    wl, ww = (-600, 1500) if args.mode=="lung" else (40, 400)
    src = Path(args.src); dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for p in sorted(src.iterdir()):
        if p.is_dir(): continue
        if is_dicom(p):
            hu  = read_dicom_hu(p)
            img = window_hu(hu, wl, ww)
        else:
            g   = read_gray01(p)
            img = window_img01(g)  # emulate window for PNGs

        deg = degrade_low_contrast(img, gamma_c=1.25, lam=50, sigma=0.008)
        # Save as .npy for lossless processing (safe for metrics)
        out = dst / (p.stem + ".npy")
        np.save(out, deg)
    print("Synthetic degraded set saved to", args.dst)

if __name__ == "__main__":
    main()


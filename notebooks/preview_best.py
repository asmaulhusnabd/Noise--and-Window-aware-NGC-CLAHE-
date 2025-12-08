#!/usr/bin/env python

import os
import sys

# --- Make project root importable as a package root (so 'src' works) ---
HERE = os.path.dirname(os.path.abspath(__file__))          # .../ct-contrast-nw-gc-clahe/notebooks
ROOT = os.path.abspath(os.path.join(HERE, ".."))           # .../ct-contrast-nw-gc-clahe
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pydicom, imageio.v2 as iio

from src.metrics.uiqi import uiqi
from src.metrics.ssim_wrap import ssim01      # SSIM on [0,1]
from src.metrics.fsim import fsim            # FSIM on [0,1]

# --- loader for real image -> [0,1] with brain windowing (for metrics only) ---
BRAIN_WL, BRAIN_WW = 40, 400  # use (50,130) for subdural if needed

def load_real_windowed(path, wl=BRAIN_WL, ww=BRAIN_WW):
    p = Path(path)
    if p.suffix.lower() == ".dcm":
        ds = pydicom.dcmread(str(p))
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = slope * arr + inter
        lo, hi = wl - ww/2.0, wl + ww/2.0
        x = np.clip(hu, lo, hi)
        return (x - lo) / (hi - lo + 1e-8)
    else:
        img = iio.imread(str(p))
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(np.float32)
        if img.max() > 1:
            img /= 255.0
        lo, hi = np.percentile(img, 2), np.percentile(img, 98)
        x = np.clip(img, lo, hi)
        return (x - lo) / (hi - lo + 1e-8)

def main():
    # --- paths relative to project root ---
    p_ref   = Path("data/real")
    p_synth = Path("data/synth")
    p_out   = Path("data/outputs")

    print("Found in data/real:", len(list(p_ref.iterdir())))
    print("Found in data/synth:", len(list(p_synth.iterdir())))
    print("Found in data/outputs:", len(list(p_out.iterdir())))

    # -------------------------------------------------------
    # 1) First pass: compute ΔUIQI, ΔSSIM, ΔFSIM and combined score
    # -------------------------------------------------------
    scores = []  # (stem, combined, ΔUIQI, ΔSSIM, ΔFSIM)
    num_tried = 0

    for r in sorted(p_ref.iterdir()):
        if r.is_dir() or r.name.startswith("."):
            continue
        stem = r.stem

        synth_path = p_synth / f"{stem}.npy"
        ngc_path   = p_out / f"{stem}_ngcclahe.npy"
        prop_path  = p_out / f"{stem}_proposed.npy"

        if not (synth_path.exists() and ngc_path.exists() and prop_path.exists()):
            # uncomment this if you want to debug missing stems
            # print(f"skip {stem}: some outputs missing")
            continue

        try:
            ref01 = load_real_windowed(r)
            synth = np.load(synth_path).astype(np.float32)
            ngc   = np.load(ngc_path).astype(np.float32)
            prop  = np.load(prop_path).astype(np.float32)
        except Exception as e:
            print(f"skip {r.name} in metrics pass: {e}")
            continue

        num_tried += 1

        for x in (ref01, synth, ngc, prop):
            x[x < 0.0] = 0.0
            x[x > 1.0] = 1.0

        u_ngc  = uiqi(ref01, ngc)
        s_ngc  = ssim01(ref01, ngc)
        f_ngc  = fsim(ref01, ngc)

        u_prop = uiqi(ref01, prop)
        s_prop = ssim01(ref01, prop)
        f_prop = fsim(ref01, prop)

        d_u = u_prop - u_ngc
        d_s = s_prop - s_ngc
        d_f = f_prop - f_ngc

        combined = d_u + d_s + d_f

        scores.append((stem, combined, d_u, d_s, d_f))

    print(f"\nSlices with all needed files & metrics: {num_tried}")
    print(f"Slices with valid score entries: {len(scores)}")

    if not scores:
        print("\nNo slices collected. Likely reasons:\n"
              "  - stems in data/real don't match data/synth / data/outputs\n"
              "  - *_ngcclahe.npy / *_proposed.npy / *_clahe.npy not generated for these stems\n")
        return

    scores.sort(key=lambda t: t[1], reverse=True)
    top8 = scores[:45]

    print("\nTop 8 slices by Δ(UIQI + SSIM + FSIM):")
    for stem, comb, d_u, d_s, d_f in top8:
        print(f"{stem}: combined={comb:.4f}, ΔUIQI={d_u:.4f}, "
              f"ΔSSIM={d_s:.4f}, ΔFSIM={d_f:.6f}")

    # ---------------------------------------------------
    # 2) Plot only those top-8 stems: Synthetic, CLAHE, NGC, Proposed
    # ---------------------------------------------------
    for stem, comb, d_u, d_s, d_f in top8:
        try:
            synth = np.load(p_synth / f"{stem}.npy").astype(np.float32)
            cla   = np.load(p_out / f"{stem}_clahe.npy").astype(np.float32)
            ngc   = np.load(p_out / f"{stem}_ngcclahe.npy").astype(np.float32)
            prop  = np.load(p_out / f"{stem}_proposed.npy").astype(np.float32)
        except Exception as e:
            print(f"skip {stem} in preview pass: {e}")
            continue

        for x in (synth, cla, ngc, prop):
            x[x < 0.0] = 0.0
            x[x > 1.0] = 1.0

        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(synth, cmap='gray'); ax[0].set_title("Synthetic\n(low contrast)")
        ax[1].imshow(cla,   cmap='gray'); ax[1].set_title("CLAHE")
        ax[2].imshow(ngc,   cmap='gray'); ax[2].set_title("NGC-CLAHE")
        ax[3].imshow(prop,  cmap='gray'); ax[3].set_title("Proposed\n(NW-NGC-CLAHE)")

        for a in ax:
            a.axis('off')

        fig.suptitle(
            f"{stem} | ΔUIQI={d_u:.4f}, ΔSSIM={d_s:.4f}, ΔFSIM={d_f:.6f}",
            fontsize=10
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

import argparse, numpy as np
from pathlib import Path
from src.metrics.uiqi import uiqi
from src.metrics.ssim_wrap import ssim01
from src.metrics.fsim import fsim
from src.io.dicom_png import is_dicom, read_dicom_hu, read_gray01, window_hu, window_img01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/real", help="folder of reference (clean) images")
    ap.add_argument("--out", default="data/outputs", help="folder of outputs (npy)")
    ap.add_argument("--mode", default="soft", choices=["soft","lung"], help="window preset")
    args = ap.parse_args()

    wl, ww = (-600, 1500) if args.mode=="lung" else (40, 400)
    ref = Path(args.ref); out = Path(args.out)

    rows = []
    for r in sorted(ref.iterdir()):
        if r.is_dir(): continue
        stem = r.stem
        # load reference (map to [0,1] same way as processing)
        if is_dicom(r):
            hu  = read_dicom_hu(r)
            ref01 = window_hu(hu, wl, ww)
        else:
            g    = read_gray01(r)
            ref01 = window_img01(g)

        # outputs saved as npy with synth stem; account for synth naming
        # we used synth names = real.stem.npy, so stems should match
        p_base = out / f"{stem}_ngcclahe.npy"
        p_prop = out / f"{stem}_proposed.npy"
        if not (p_base.exists() and p_prop.exists()): 
            continue

        base = np.load(p_base); prop = np.load(p_prop)

        u_base = uiqi(ref01, base);  s_base = ssim01(ref01, base);  f_base = fsim(ref01, base)
        u_prop = uiqi(ref01, prop);  s_prop = ssim01(ref01, prop);  f_prop = fsim(ref01, prop)

        rows.append((stem, u_base, s_base, f_base, u_prop, s_prop, f_prop))

    # Print a simple table
    if rows:
        print("stem, UIQI_base, SSIM_base, FSIM_base, UIQI_prop, SSIM_prop, FSIM_prop")
        for t in rows: print(",".join([t[0]] + [f"{x:.4f}" for x in t[1:]]))
        # Means
        mean = lambda k: sum(r[k] for r in rows)/len(rows)
        print("\nMeans over", len(rows), "images:")
        print("Base    UIQI/SSIM/FSIM =", f"{mean(1):.4f}", f"{mean(2):.4f}", f"{mean(3):.4f}")
        print("Proposed UIQI/SSIM/FSIM =", f"{mean(4):.4f}", f"{mean(5):.4f}", f"{mean(6):.4f}")
    else:
        print("No rows computed. Check that names in data/real match outputs in data/outputs.")

if __name__ == "__main__":
    main()


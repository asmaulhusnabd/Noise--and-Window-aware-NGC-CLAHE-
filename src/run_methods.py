import argparse, numpy as np
from pathlib import Path
from src.io.dicom_png import is_dicom, read_dicom_hu, read_gray01, window_hu, window_img01
from src.enhan.ngc_clahe import ngc_clahe
from src.enhan.nw_gc_clahe import nw_gc_clahe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/synth", help="images (npy for synth, or PNG/DICOM)")
    ap.add_argument("--out", default="data/outputs", help="save folder (npy)")
    ap.add_argument("--mode", default="soft", choices=["soft","lung"], help="window preset")
    args = ap.parse_args()

    wl, ww = (-600, 1500) if args.mode=="lung" else (40, 400)
    src = Path(args.src); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for p in sorted(src.iterdir()):
        if p.is_dir(): continue

        # Load input (synth uses npy; real can be DICOM/PNG)
        if p.suffix.lower()==".npy":
            x = np.load(p)
            img01 = x
        elif is_dicom(p):
            hu  = read_dicom_hu(p)
            img01 = window_hu(hu, wl, ww)
        else:
            g   = read_gray01(p)
            img01 = window_img01(g)

        # Baseline (NGC-CLAHE)
        base = ngc_clahe(img01, gamma=0.95, clip=2.0, tile=(8,8))

        # Proposed (NW-GC-CLAHE)
        prop, _ = nw_gc_clahe(img01, gamma=0.95, clip_cons=1.0, clip_agg=3.0, tile=(8,8))

        np.save(out / (p.stem + "_ngcclahe.npy"), base)
        np.save(out / (p.stem + "_proposed.npy"), prop)

    print("Saved outputs to", args.out)

if __name__ == "__main__":
    main()


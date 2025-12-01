import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import pydicom, imageio.v2 as iio
import numpy as np

# --- simple loader for real image -> [0,1] with brain windowing ---
BRAIN_WL, BRAIN_WW = 40, 80  # use (50,130) for subdural

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
        if img.ndim == 3: img = img[...,0]
        img = img.astype(np.float32)
        if img.max() > 1: img /= 255.0
        # percentile “window” for PNG/JPG
        lo, hi = np.percentile(img, 2), np.percentile(img, 98)
        x = np.clip(img, lo, hi)
        return (x - lo) / (hi - lo + 1e-8)

# --- your preview, now with 3 panels (Real, NGC-CLAHE, Proposed) ---
p_ref = Path("data/real")
p_out = Path("data/outputs")

for r in sorted(p_ref.iterdir()):
    if r.is_dir(): 
        continue
    stem = r.stem
    try:
        real01 = load_real_windowed(r)  # << new
        base = np.load(p_out / f"{stem}_ngcclahe.npy")
        prop = np.load(p_out / f"{stem}_proposed.npy")
    except Exception as e:
        print(f"skip {r.name}: {e}")
        continue

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(real01, cmap='gray'); ax[0].set_title(f"Original (WL/WW {BRAIN_WL}/{BRAIN_WW})")
    ax[1].imshow(base,   cmap='gray'); ax[1].set_title("NGC-CLAHE")
    ax[2].imshow(prop,   cmap='gray'); ax[2].set_title("Proposed (NW-GC-CLAHE)")
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.show()

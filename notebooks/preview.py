import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

p_ref = Path("data/real")
p_out = Path("data/outputs")

for r in sorted(p_ref.iterdir()):
    if r.is_dir(): continue
    stem = r.stem
    try:
        base = np.load(p_out / f"{stem}_ngcclahe.npy")
        prop = np.load(p_out / f"{stem}_proposed.npy")
    except:
        continue
    fig,ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].imshow(base, cmap='gray'); ax[0].set_title("NGC-CLAHE")
    ax[1].imshow(prop, cmap='gray'); ax[1].set_title("Proposed (NW-GC-CLAHE)")
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.show()


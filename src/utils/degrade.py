import numpy as np
from scipy.ndimage import gaussian_filter

def degrade_low_contrast(img01,
                         # --- contrast controls (main effect) ---
                         gamma_c=1.15,          # 1.1–1.3 makes mids flatter (no huge darkening)
                         dyn_range=0.65,        # 0.55–0.75 keeps a narrower range
                         mid_shift=0.18,        # 0.10–0.25 recenters the range
                         # --- noise controls (subtle) ---
                         lam=1200,              # 800–2000 ⇒ very light Poisson noise
                         sigma=0.002,           # 0.001–0.004 small Gaussian noise
                         # --- optional mild blur ---
                         blur_sigma=0.4):       # 0.3–0.6 slight smoothing

    x = np.clip(img01, 0, 1)

    # 1) Contrast compression (dominant)
    #    Flatten mid-tones and shrink dynamic range like the paper’s “low contrast reduction”.
    x = x ** float(gamma_c)
    x = mid_shift + dyn_range * x
    x = np.clip(x, 0, 1)

    # 2) Very light Poisson noise (high counts -> subtle noise)
    if lam is not None and lam > 0:
        counts = x * lam
        x = np.random.poisson(counts).astype(np.float32) / float(lam)

    # 3) Tiny Gaussian noise
    if sigma and sigma > 0:
        x = x + np.random.normal(0.0, float(sigma), x.shape).astype(np.float32)

    # 4) Optional slight blur (mimics mild reconstruction smoothing)
    if blur_sigma and blur_sigma > 0:
        x = gaussian_filter(x, float(blur_sigma))

    return np.clip(x, 0, 1)

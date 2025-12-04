import numpy as np

def degrade_low_contrast(img01, strength: str = "medium") -> np.ndarray:
    """
    Simulate a low-contrast CT slice without adding synthetic noise.

    This function is designed to mimic the contrast reduction used in the
    NGC-CLAHE paper for synthetic experiments:
      - We work on the *windowed* image in [0,1].
      - We shrink the dynamic range around the global mean (linear compression).
      - We add a gentle gamma (>1) for a nonlinear 'flattening'.
      - No salt-and-pepper or extra noise is added.

    Parameters
    ----------
    img01 : np.ndarray
        Input image as float in [0,1]. This should be the output of
        window_hu(...) or window_img01(...).
    strength : {"mild", "medium", "strong"}, optional
        Amount of contrast reduction:
          * "mild"   – small flattening (closer to original)
          * "medium" – default, similar to paper's examples
          * "strong" – heavier contrast loss

    Returns
    -------
    np.ndarray
        Degraded image as float32 in [0,1].
    """
    # ensure float and clipped
    x = np.clip(img01.astype(np.float32), 0.0, 1.0)

    # choose parameters for each severity
    if strength == "mild":
        # slight shrink, tiny gamma
        shrink_factor = 0.8
        gamma = 1.10
    elif strength == "strong":
        # strong shrink, stronger gamma
        shrink_factor = 0.4
        gamma = 1.35
    else:  # "medium"
        shrink_factor = 0.6
        gamma = 1.25

    # 1) linear dynamic-range compression around the mean
    mu = float(x.mean())
    y = (x - mu) * shrink_factor + mu
    y = np.clip(y, 0.0, 1.0)

    # 2) gentle nonlinear compression (keeps look closer to paper)
    y = np.power(y, gamma)
    y = np.clip(y, 0.0, 1.0)

    return y.astype(np.float32)

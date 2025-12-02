import numpy as np
import cv2

def clahe_baseline(img01: np.ndarray,
                   clip: float = 2.0,
                   tile=(8, 8)) -> np.ndarray:
    """
    Plain CLAHE baseline on a windowed image in [0,1].

    Parameters
    ----------
    img01 : np.ndarray
        Input image, float32 in [0,1].
    clip : float
        CLAHE clip limit.
    tile : (int, int)
        Tile grid size.

    Returns
    -------
    np.ndarray
        Enhanced image, float32 in [0,1].
    """
    x = np.clip(img01.astype(np.float32), 0.0, 1.0)
    u8 = np.uint8(np.round(x * 255.0))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    out = clahe.apply(u8)
    return (out.astype(np.float32) / 255.0)

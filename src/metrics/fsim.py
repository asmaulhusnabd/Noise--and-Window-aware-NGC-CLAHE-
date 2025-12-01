import numpy as np
from skimage.filters import scharr

def fsim(img1, img2, T1=0.85, T2=160.0):
    # expects [0,1]
    i1 = img1.astype(np.float32)
    i2 = img2.astype(np.float32)

    # Gradient magnitude as a proxy feature (simple FSIM variant)
    G1 = np.abs(scharr(i1))
    G2 = np.abs(scharr(i2))

    PC1 = G1 / (G1.max()+1e-8)
    PC2 = G2 / (G2.max()+1e-8)

    S_pc = (2*PC1*PC2 + T1) / (PC1*PC1 + PC2*PC2 + T1)
    S_g  = (2*G1*G2 + T2) / (G1*G1 + G2*G2 + T2)

    # weight more where feature energy is high
    W = np.maximum(PC1, PC2)
    FSIM_map = (S_pc * S_g)

    num = (FSIM_map * W).sum()
    den = (W.sum() + 1e-8)
    return float(num / den)


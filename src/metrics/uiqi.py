import numpy as np
from scipy.ndimage import uniform_filter

def uiqi(img1, img2, win_size=8):
    # expects [0,1]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    K = win_size
    mu1 = uniform_filter(img1, K)
    mu2 = uniform_filter(img2, K)
    mu1_sq, mu2_sq = mu1*mu1, mu2*mu2
    mu12 = mu1*mu2

    sigma1_sq = uniform_filter(img1*img1, K) - mu1_sq
    sigma2_sq = uniform_filter(img2*img2, K) - mu2_sq
    sigma12   = uniform_filter(img1*img2, K) - mu12

    numerator   = 4 * mu12 * sigma12
    denominator = (mu1_sq + mu2_sq) * (sigma1_sq + sigma2_sq)
    qmap = (numerator + 1e-8) / (denominator + 1e-8)
    return float(np.mean(qmap))


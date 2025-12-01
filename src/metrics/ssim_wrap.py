import numpy as np
from skimage.metrics import structural_similarity as ssim

def ssim01(img1, img2):
    # expects [0,1]
    return float(ssim(img1.astype(np.float32),
                      img2.astype(np.float32),
                      data_range=1.0))


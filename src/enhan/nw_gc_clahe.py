import numpy as np, cv2
from skimage.filters import sobel
from scipy.ndimage import uniform_filter
from .ngc import ngc

def clahe01(img01, clip=2.0, tile=(8,8)):
    u8 = np.uint8(np.clip(img01*255, 0, 255))
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(u8)
    return out.astype(np.float32)/255.0

def edge_map(img01):
    e = np.abs(sobel(img01))
    e = (e - e.min()) / (e.max()-e.min()+1e-8)
    return e

def noise_map(img01, k=7, edge=None):
    m  = uniform_filter(img01, size=k)
    m2 = uniform_filter(img01*img01, size=k)
    var = np.maximum(m2 - m*m, 0.0)
    z = var
    if edge is not None:
        z = z * (1.0 - edge)  # emphasize flat-noisy
    z = (z - z.min())/(z.max()-z.min()+1e-8)
    return z

def nw_gc_clahe(img01, gamma=0.95,
                clip_cons=1.0, clip_agg=3.0, tile=(8,8),
                alpha=0.8, beta=0.6, delta=0.2):
    x = ngc(img01, gamma=gamma)
    E = edge_map(x)
    N = noise_map(x, k=7, edge=E)
    cons = clahe01(x, clip=clip_cons, tile=tile)
    agg  = clahe01(x, clip=clip_agg,  tile=tile)
    W = np.clip(alpha*E - beta*N + delta, 0.0, 1.0)
    out = W*agg + (1.0-W)*cons
    return out, (E, N, W)


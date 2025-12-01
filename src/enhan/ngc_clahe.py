import numpy as np, cv2
from .ngc import ngc

def clahe01(img01, clip=2.0, tile=(8,8)):
    u8 = np.uint8(np.clip(img01*255, 0, 255))
    out = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(u8)
    return out.astype(np.float32)/255.0

def ngc_clahe(img01, gamma=0.95, clip=2.0, tile=(8,8)):
    x = ngc(img01, gamma=gamma)
    return clahe01(x, clip=clip, tile=tile)


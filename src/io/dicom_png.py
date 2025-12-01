import numpy as np
import imageio.v2 as iio
import pydicom
from pathlib import Path

def read_dicom_hu(path):
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = slope * arr + inter
    return hu

def read_gray01(path):
    # For PNG/JPG: return [0,1]
    img = iio.imread(str(path))
    if img.ndim == 3:
        img = img[...,0]
    img = img.astype(np.float32)
    if img.max() > 1.0: img /= 255.0
    return np.clip(img, 0, 1)

def is_dicom(path: Path):
    return path.suffix.lower() == ".dcm"

def window_hu(hu, wl, ww):
    lo, hi = wl - ww/2.0, wl + ww/2.0
    x = np.clip(hu, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)

def window_img01(img01, p_lo=2, p_hi=98):
    # If you only have PNGs and no HU, emulate a window by percentiles
    lo = np.percentile(img01, p_lo)
    hi = np.percentile(img01, p_hi)
    x = np.clip(img01, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)


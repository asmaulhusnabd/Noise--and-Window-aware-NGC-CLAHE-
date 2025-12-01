import numpy as np

def ngc(img01, gamma=0.95):
    g = np.clip(img01, 0, 1) ** float(gamma)
    gmin, gmax = g.min(), g.max()
    return (g - gmin) / (gmax - gmin + 1e-8)


#!/opt/anaconda3/envs/p11/bin/python3

# -*- coding: utf-8 -*-
# minimal zscale + quantize utilities
from astropy.io import fits
from PIL import Image
import os
import numpy as np
from astropy.visualization import ZScaleInterval  # type: ignore


__all__ = ["zscale_window", "to_int_image", "autoscale_to_int", "save_8bit_image"]


def zscale_window(data, nsamples=1000, contrast=0.25):
    """Return (vmin, vmax) by zscale; fallback to percentiles if astropy missing."""
    flat = np.asarray(data)
    flat = flat[np.isfinite(flat)].ravel()
    if flat.size == 0:
        raise ValueError("no finite values")
    vmin, vmax = ZScaleInterval(nsamples=nsamples, contrast=contrast).get_limits(flat)
    return float(vmin), float(vmax)


essential_max = {8: 255.0, 16: 65535.0}

def to_int_image(data, vmin, vmax, *, bits=16, clip=True, gamma=None):
    if bits not in essential_max:
        raise ValueError("bits must be 8 or 16")
    d = np.asarray(data, dtype=np.float64)
    d = np.where(np.isfinite(d), d, vmin)
    if clip:
        d = np.clip(d, vmin, vmax)
    s = (d - vmin) / (vmax - vmin)
    if gamma and gamma > 0:
        s = s ** (1.0 / float(gamma))
    scale = essential_max[bits]
    return (s * scale + 0.5).astype(np.uint8 if bits == 8 else np.uint16)


def autoscale_to_int(data, *, bits=8, nsamples=1000, contrast=0.25, clip=True, gamma=None):
    vmin, vmax = zscale_window(data, nsamples=nsamples, contrast=contrast)
    return to_int_image(data, vmin, vmax, bits=bits, clip=clip, gamma=gamma), vmin, vmax



def save_8bit_image(data, path):
    scaled, vmin, vmax = autoscale_to_int(data, bits=8)
    Image.fromarray(scaled, mode="L").save(path)



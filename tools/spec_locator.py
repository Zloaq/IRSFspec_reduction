#!/opt/anaconda3/envs/p11/bin/python3

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import label
from pathlib import Path


def find_dark(fitsname, darkpath):
    fits_basename = Path(fitsname).name
    m = re.match(r"(CDS\d{2})", fits_basename)
    if not m:
        raise ValueError(f"Could not extract CDS number from filename: {fits_basename}")
    cds_num = m.group(1)

    pattern = str(Path(darkpath) / f"*{cds_num}.fits")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No dark frame found for {cds_num} in {darkpath}")

    return fits.getdata(matches[0])


def calc_sigma(fits_data):

    center_x, center_y = 300, 30
    width, height = 500, 30

    rect = fits_data[
        center_y - height // 2 : center_y + height // 2,
        center_x - width // 2  : center_x + width // 2
    ]
    median = np.median(rect)
    rms = np.sqrt(np.mean((rect - median) ** 2))

    return {"median": median, "stddev": rms}

def mask_outside_band_to_zero(data: np.ndarray, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    """
    2本の直線 y=a1*x+b1 と y=a2*x+b2 に挟まれた帯域『以外』の画素を 0 にする。
    data は (行=Y, 列=X) の2次元配列を想定。
    """
    h, w = data.shape
    # Y: (h,1), X: (1,w) を作る（メモリ効率の良い ogrid を使用）
    Y, X = np.ogrid[:h, :w]
    l1 = a1 * X + b1
    l2 = a2 * X + b2
    lower = np.minimum(l1, l2)  # 各x列での下側境界
    upper = np.maximum(l1, l2)  # 各x列での上側境界

    # 帯域『外』= 下側より下 or 上側より上
    mask_out = (Y < lower) | (Y > upper)

    out = data.copy()
    out[mask_out] = 0
    return out


def mask_region_to_zero(data: np.ndarray) -> np.ndarray:
    """元コードの (a,b) を踏襲し、帯域の外側を 0 にする。"""
    return mask_outside_band_to_zero(data, a1=2.3, b1=-50, a2=2.5, b2=-1100)


def data_mask(data, sky_noise_data, threshold=1.0):
    thr = sky_noise_data["median"] - threshold * sky_noise_data["stddev"]
    return data <= thr



def find_box(data, sky_noise_data, threshold=1, min_area=200):


    mask = data_mask(data, sky_noise_data, threshold)
    labeled, num_features = label(mask.astype(np.uint8))
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    areas = np.bincount(labeled.ravel())[1:]
    valid_labels = np.where(areas >= min_area)[0] + 1  # ラベル番号は1始まり

    final_labels = []
    for label_id in valid_labels:
        coords = np.argwhere(labeled == label_id)
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        x_span = x_coords.max() - x_coords.min() + 1
        y_span = y_coords.max() - y_coords.min() + 1
        if y_span < 25 and x_span > 25:
            final_labels.append(label_id)

    combined_mask = np.isin(labeled, final_labels)
    return combined_mask



def main(fits_data_iterator, DARK_PATH):
    combined_masks = []
    for fits_path in fits_data_iterator:
        data = fits.getdata(fits_path)
        dark_data = find_dark(fits_path, DARK_PATH)
        data = data - dark_data
        sky_noise_data = calc_sigma(data)
        data = mask_region_to_zero(data)
        for threshold in np.arange(4.0, 0.99, -0.5):
            combined_mask = find_box(data, sky_noise_data, threshold)
            if combined_mask.sum() > 0:
                break
        if combined_mask.sum() == 0:
            combined_masks.append(None)
            continue
        combined_masks.append(combined_mask)
    return combined_masks
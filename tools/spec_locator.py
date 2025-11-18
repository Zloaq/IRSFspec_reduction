#!/opt/anaconda3/envs/p11/bin/python3

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import label
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceConfig:
    # calc_sigma window
    center_x: int = 300
    center_y: int = 30
    width: int = 500
    height: int = 30

    # band lines for mask_outside_band_to_zero
    band_a1: float = 2.3
    band_b1: float = -50.0
    band_a2: float = 2.5
    band_b2: float = -1100.0

    # find_box geometry filters
    min_area: int = 200
    y_span_lt: int = 25   # keep condition: y_span < 25
    x_span_gt: int = 25   # keep condition: x_span > 25

    # thresholds for spec_locator loop
    thresholds: tuple = tuple(np.arange(4.0, 0.99, -0.5))

DEFAULT_CONFIG = DeviceConfig()


def calc_sigma(fits_data, cfg: DeviceConfig = DEFAULT_CONFIG):
    center_x, center_y = cfg.center_x, cfg.center_y
    width, height = cfg.width, cfg.height

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


def mask_region_to_zero(data: np.ndarray, cfg: DeviceConfig = DEFAULT_CONFIG) -> np.ndarray:
    """帯域の外側を 0 にする（装置依存パラメータは cfg から）。"""
    return mask_outside_band_to_zero(
        data,
        a1=cfg.band_a1, b1=cfg.band_b1,
        a2=cfg.band_a2, b2=cfg.band_b2
    )


def data_mask(data, sky_noise_data, threshold=1.0):
    thr = sky_noise_data["median"] - threshold * sky_noise_data["stddev"]
    return data <= thr


def find_box(data, sky_noise_data, threshold=1, cfg: DeviceConfig = DEFAULT_CONFIG):

    mask = data_mask(data, sky_noise_data, threshold)
    labeled, num_features = label(mask.astype(np.uint8))
    if num_features == 0:
        return np.zeros_like(mask, dtype=bool)

    areas = np.bincount(labeled.ravel())[1:]
    valid_labels = np.where(areas >= cfg.min_area)[0] + 1  # ラベル番号は1始まり

    final_labels = []
    for label_id in valid_labels:
        coords = np.argwhere(labeled == label_id)
        x_coords = coords[:, 1]
        y_coords = coords[:, 0]
        x_span = x_coords.max() - x_coords.min() + 1
        y_span = y_coords.max() - y_coords.min() + 1
        if y_span < cfg.y_span_lt and x_span > cfg.x_span_gt:
            final_labels.append(label_id)

    combined_mask = np.isin(labeled, final_labels)
    return combined_mask



def spec_locator(data, cfg: DeviceConfig = DEFAULT_CONFIG):

    sky_noise_data = calc_sigma(data, cfg)
    data = mask_region_to_zero(data, cfg)
    combined_mask = np.zeros_like(data, dtype=bool)
    for threshold in cfg.thresholds:
        combined_mask = find_box(data, sky_noise_data, threshold, cfg)
        #print(f"Threshold {threshold} found {combined_mask.sum()} pixels")
        if combined_mask is not None and combined_mask.sum() > 0:
            break
    if combined_mask is None or combined_mask.sum() == 0:
        return None
    return combined_mask


if __name__ == "__main__":

    import glob
    import astropy.io.fits as fits
    import re
    import os
    import logging
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(f"../config.env")
    RAWDATA_DIR = os.getenv("RAWDATA_DIR")
    DARK4LOCATE_DIR = os.getenv("DARK4LOCATE_DIR")
    WORK_DIR = os.getenv("WORK_DIR")
    fitslist = glob.glob(f"{RAWDATA_DIR}/s-gem/220120/*-0094*.fits")

    fitslist_sorted = sorted(fitslist, reverse=True)
    print(RAWDATA_DIR)
    print(fitslist_sorted)

    def find_dark(fitsname, darkpath):
        fits_basename = Path(fitsname).name
        m = re.search(r"(CDS\d{2})", fits_basename)
        if not m:
            raise ValueError(f"Could not extract CDS number from filename: {fits_basename}")
        cds_num = m.group(1)

        pattern = str(Path(darkpath) / f"*{cds_num}.fits")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No dark frame found for {cds_num} in {darkpath}")

        return fits.getdata(matches[0])

    center_y = None
    num = 0

    for fits_path in fitslist_sorted:
        header = fits.getheader(fits_path)
        data = fits.getdata(fits_path)
        dark = find_dark(fits_path, DARK4LOCATE_DIR)
        image = data - dark
        mask = spec_locator(image)
        if mask is None:
            # このファイルではスペクトルが見つからなかった → 次のファイルへ
            logging.warning(f"spec_locator failed for {os.path.basename(fits_path)}: skipping.")
            fig, ax = plt.subplots()
            ax.imshow(image)
            plt.savefig(f"{WORK_DIR}/test{num}_none.png")
            num += 1
            continue
        else:
            # このファイルではスペクトルが見つかった
            fig, ax = plt.subplots()
            ax.imshow(mask)
            plt.savefig(f"{WORK_DIR}/test{num}.png")
            num += 1
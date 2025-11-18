#!/opt/anaconda3/envs/p11/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class SpecConfig:
    spec_spread_pix:int = 12
    buffer_pix:int = 30



# --- AB classification helpers ---
def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    True成分を最小で囲む矩形(Bounding Box)を返す。
    Returns: (ymin, ymax, xmin, xmax)
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("mask has no True pixels")
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def vertical_center_from_mask(mask: np.ndarray) -> float:
    """
    矩形の縦方向中心(y)を返す。floatで返す。
    """
    ymin, ymax, _, _ = bbox_from_mask(mask)
    return int(0.5 * (ymin + ymax))


def classify_spec_location(centerdict: Dict[str, int], config: SpecConfig = SpecConfig()) -> Dict[str, str]:
    """
    入力: {No: y_center} の辞書。
    分離判定: 連続する中心の差 (dy) >  2*spec_spread_pix + buffer_pix で新しい群とみなす。
    y が小さい順に A, B, C, ... を割り当てて返す。

    Returns: {No: label}
    """
    # y(center) で昇順ソート
    items = sorted(centerdict.items(), key=lambda kv: float(kv[1]))

    # 分離のしきい値
    threshold = 2 * float(config.spec_spread_pix) + float(config.buffer_pix)

    groups: List[List[tuple[Any, float]]] = []
    current: List[tuple[Any, float]] = [items[0]]

    # 隣接する中心の差で逐次クラスタリング
    for prev, cur in zip(items, items[1:]):
        dy = float(cur[1]) - float(prev[1])
        if dy > threshold:
            groups.append(current)
            current = [cur]
        else:
            current.append(cur)
    groups.append(current)

    def idx_to_label(idx: int) -> str:
        """0->'A', 1->'B', ... 25->'Z', 26->'AA' という Excel 風ラベル。"""
        idx0 = int(idx)
        label = ""
        while True:
            label = chr(ord('A') + (idx0 % 26)) + label
            idx0 = idx0 // 26 - 1
            if idx0 < 0:
                break
        return label

    result: Dict[Any, str] = {}
    for gi, group in enumerate(groups):
        lab = idx_to_label(gi)
        for k, _y in group:
            result[k] = lab
    return result


if __name__ == "__main__":

    import glob
    import astropy.io.fits as fits
    import re
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    import spec_locator
    import logging
    import matplotlib.pyplot as plt


    load_dotenv("../config.env")
    RAWDATA_DIR = os.getenv("RAWDATA_DIR")
    DARK4LOCATE_DIR = os.getenv("DARK4LOCATE_DIR")
    WORK_DIR = os.getenv("WORK_DIR")
    fitslist = glob.glob(f"{RAWDATA_DIR}/s-gem/220120/*-0091*.fits")

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


    fitslist_sorted = sorted(fitslist, reverse=True)
    print(RAWDATA_DIR)
    print(fitslist_sorted)

    center_y = None

    for fits_path in fitslist_sorted:
        header = fits.getheader(fits_path)
        data = fits.getdata(fits_path)
        dark = find_dark(fits_path, DARK4LOCATE_DIR)
        image = data - dark
        mask = spec_locator.spec_locator(image)
        if mask is None:
            # このファイルではスペクトルが見つからなかった → 次のファイルへ
            logging.warning(f"spec_locator failed for {os.path.basename(fits_path)}: skipping.")
            continue

        # 最初に見つかった mask を採用してループを抜ける
        center_y = vertical_center_from_mask(mask)
        break
    

    fig, ax = plt.subplots()
    ax.imshow(mask)
    plt.savefig(f"{WORK_DIR}/test.png")
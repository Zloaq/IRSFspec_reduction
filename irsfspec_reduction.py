#!/opt/anaconda3/envs/p11/bin/python3

from dotenv import load_dotenv
import os
import sys
import sqlite3
import subprocess
import logging
from pathlib import Path
import re
import glob
import astropy.io.fits as fits
import numpy as np
from typing import Dict, List, Set, Tuple

import spec_locator
import classify_spec_location as csl

load_dotenv("config.env")

DB_PATH = os.getenv("DB_PATH")
RAW_DARK_PATH = os.getenv("RAW_DARK_PATH")
DARK4LOCATE_DIR = os.getenv("DARK4LOCATE_DIR")
RAID_PC = os.getenv("RAID_PC")
RAID_DIR = os.getenv("RAID_DIR")
DST_DIR = os.getenv("DST_DIR")

QUALITY_HIST_RANGE = (55000, 65536)
BINS = 1000
SATURATION_LEVEL = 18000



def _load_fits(fits_path: str) -> Tuple[fits.Header, np.ndarray]:
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS not found: {fits_path}")
    with fits.open(fits_path, memmap=False) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    if data is None:
        raise ValueError(f"No data in primary HDU: {fits_path}")
    data = np.asarray(data, dtype=np.float64)
    return header, data


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


def db_search(conn: sqlite3.Connection, object_name, date_label):
    
    """
    framesテーブルからAr系のframeを抽出し、
    {date_label: [base_name, ...]} の辞書を返す。
    """
    query = (
        "SELECT date_label, base_name "
        "FROM frames "
        f"WHERE object LIKE '{object_name}' "
    )
    cur = conn.cursor()
    cur.execute(query)

    filepath_dict: Dict[str, List[str]] = {}
    for date_label, base_name in cur.fetchall():
        filepath_dict.setdefault(date_label, []).append(base_name)
    return filepath_dict
    

def do_scp(date_label: str, Number: int) -> None:
    src = f"{RAID_PC}:{RAID_DIR}/{date_label}/spec/*{Number:04}*CDS*.fits"
    dst = DST_DIR
    cmd = ["scp", "-r", src, dst]

    logging.info(f"COPY: {src}")
    subprocess.run(cmd, check=True)


def compute_hist(data: np.ndarray, bins: int = BINS, rng=QUALITY_HIST_RANGE):
    """メモリ節約のため、事前に mask を作らずに全体から直接ヒストグラムを計算する。
    範囲外(<=rng[0] や >rng[1])の値は np.histogram の仕様で自動的に無視される。
    """
    # 1Dコピーで明示的に扱う（メモリ節約しない）
    values = np.ravel(data).astype(np.float64)
    # BLANK/NaN/Infを除去
    finite = np.isfinite(values)
    values = values[finite]
    # 範囲が指定されなければデータ全体のmin/maxを使う
    if rng is None:
        vmin = float(np.min(values)) if values.size else 0.0
        vmax = float(np.max(values)) if values.size else 1.0
        rng = (vmin, vmax)
    hist, bin_edges = np.histogram(values, bins=bins, range=rng)
    return hist, bin_edges


def hist_area_from_counts(hist, bin_edges, log_hist: bool = False) -> float:
    """既に計算済みのヒストグラムから面積（bin幅×countの合計）を返す。
    log_hist=True の場合は count に log10(count+1) を適用した面積を返す。
    """
    if log_hist:
        hist = np.log10(hist + 1)
    bin_widths = np.diff(bin_edges)
    return float(np.sum(hist * bin_widths))


def compute_area(data, bins: int = BINS, rng=QUALITY_HIST_RANGE, log_hist: bool = False) -> float:
    """単一入力の面積を計算して数値を返す。
    x: ndarray もしくは FITSファイルパス(str/Path)
    """
    hist, bin_edges = compute_hist(data, bins=bins, rng=rng)
    return hist_area_from_counts(hist, bin_edges, log_hist=log_hist)


def quality_check(fitslist: List[str]):
    pass_list = []
    for fits_path in fitslist:
        header, data = _load_fits(fits_path)
        area = compute_area(data)
        if area > 0.0:
            print("read error", os.path.basename(fits_path), area)
            continue
        pass_list.append(fits_path)
    return pass_list


def classify_spec_location(fitsdict: Dict[str, List[str]]) -> Dict[str, str]:
    """
    fitsdict: {number: [fits_path, ...]}
    それぞれのfitsについて mask の True を包含する最小矩形を求め、
    その矩形の中心を計算する。
    Returns: {No: label}
    """
    result: Dict[str, Dict[str, object]] = {}

    for number, fitslist in fitsdict.items():
        centers: Dict[str, int] = {}
        files: List[str] = []
        fitslist.sort()
        for fits_path in fitslist[::-1]:
            header, data = _load_fits(fits_path)
            dark = find_dark(fits_path, DARK4LOCATE_DIR)
            image = data - dark
            mask = spec_locator.spec_locator(image)
            if mask is None:
                continue
        if mask is None:
            center_y = None
        else:
            center_y = csl.vertical_center_from_mask(mask)
        centers[number] = center_y
    result = csl.classify_spec_location(centers)

    return result


def reject_saturation(fitslist: List[str]):
    pass_fitslist = []
    for fits_path in fitslist:
        header, data = _load_fits(fits_path)
        mask = spec_locator.spec_locator(data)
        if mask is None:
            continue
        if data[mask] < SATURATION_LEVEL:
            continue
        pass_fitslist.append(fits_path)
    return pass_fitslist



def search_combination_CDSnum(fitslist1, fitslist2):
    fitslist1.sort()
    fitslist2.sort()
    cdsnum1 = [ re.match(r"CDS(\d{2})", os.path.basename(fits_path)).group(1) for fits_path in fitslist1 ]
    cdsnum2 = [ re.match(r"CDS(\d{2})", os.path.basename(fits_path)).group(1) for fits_path in fitslist2 ]
    common = set(cdsnum1) & set(cdsnum2)
    firstnum = common[0]
    lastnum = common[-1]
    fits_tuple1 = (fitslist1[cdsnum1.index(firstnum)], fitslist1[cdsnum1.index(lastnum)])
    fits_tuple2 = (fitslist2[cdsnum2.index(firstnum)], fitslist2[cdsnum2.index(lastnum)])
    return fits_tuple1, fits_tuple2



def reduction_main():
    object_name = sys.argv[1]
    date_label = sys.argv[2]
    filepath_dict = db_search(conn, object_name, date_label)
    
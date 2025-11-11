#!/opt/anaconda3/envs/p11/bin/python3

from dotenv import load_dotenv
import os
import sqlite3
import subprocess
import logging
import astropy.io.fits as fits
import numpy as np
from typing import Dict, List, Set, Tuple

import spec_locator

load_dotenv()

DB_PATH = os.getenv("DB_PATH")
RAW_DARK_PATH = os.getenv("RAW_DARK_PATH")
DARK4LOCATE_DIR = os.getenv("DARK4LOCATE_DIR")
RAID_PC = os.getenv("RAID_PC")
RAID_DIR = os.getenv("RAID_DIR")
DST_DIR = os.getenv("DST_DIR")

QUALITY_HIST_RANGE = (55000, 65536)
BINS = 1000



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


def classify_AB(fitsdict: Dict[str, List[str]]):

    for number, fitslist in fitsdict.items():
        box = []
        


def reject_saturation():
    
    pass

def search_combination_CDS_n_AB():
    pass
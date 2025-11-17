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
from itertools import combinations
from typing import Dict, List, Set, Tuple

import spec_locator
import classify_spec_location as csl

load_dotenv("config.env")

DB_PATH = os.getenv("DB_PATH")
DARK4LOCATE_DIR = os.getenv("DARK4LOCATE_DIR")
RAID_PC = os.getenv("RAID_PC")
RAID_DIR = os.getenv("RAID_DIR")
RAWDATA_DIR= os.getenv("RAWDATA_DIR")
WORK_DIR = os.getenv("WORK_DIR")

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


def db_search(conn: sqlite3.Connection, object_name, date_label=None) -> Dict[str, List[str]]:
    
    """
    framesテーブルからAr系のframeを抽出し、
    {date_label: [base_name, ...]} の辞書を返す。
    """
    if date_label is None:
        query = (
            "SELECT date_label, base_name "
            "FROM frames "
            f"WHERE object LIKE '{object_name}' "
        )
    else:
        query = (
            "SELECT date_label, base_name "
            "FROM frames "
            f"WHERE object LIKE '{object_name}' "
            f"AND date_label = '{date_label}' "
        )
    cur = conn.cursor()
    cur.execute(query)

    filepath_dict: Dict[str, List[str]] = {}
    for date_label, base_name in cur.fetchall():
        filepath_dict.setdefault(date_label, []).append(base_name)
    return filepath_dict
    

def do_scp_raw_fits(date_label: str, object_name: str, base_name_list: List[str]) -> None:
    # Extract Num1 from basenames
    num1_set = set()
    for bn in base_name_list:
        m = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}\.fits", bn)
        if m:
            num1_set.add(m.group(1))

    # If nothing matched, do nothing
    if not num1_set:
        logging.warning(f"No valid Num1 found in base_name_list for {date_label}")
        return

    # Perform scp per Num1
    for num1 in sorted(num1_set):
        src = f"{RAID_PC}:{RAID_DIR}/{date_label}/spec/spec{date_label}*{num1}*CDS*.fits"
        dst = f"{RAWDATA_DIR}/{object_name}/{date_label}"
        os.makedirs(dst, exist_ok=True)
        cmd = ["scp", src, dst]

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


def gen_fitsdict(fitslist: List[str]) -> Dict[str, List[str]]:
    fitsdict: Dict[str, List[str]] = {}
    for fits_path in fitslist:
        num1 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}\.fits", os.path.basename(fits_path)).group(1)
        fitsdict.setdefault(num1, []).append(fits_path)
    return fitsdict


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



def search_combination_with_diff_label(
        label_dict: Dict[str, str],
    ) -> List[Tuple[str, str]]:

    results: List[Tuple[str, str]] = []

    for no1, no2 in combinations(sorted(label_dict.keys()), 2):
        # ラベルが同じならスキップ
        if label_dict[no1] == label_dict[no2]:
            continue
        results.append((no1, no2))

    return results



def search_combination_for_set_AB(fitslist1: List[str], fitslist2: List[str]):
    results: List[Tuple[str, str]] = []
    cdsnum1_list = []
    cdsnum2_list = []
    for fits_path1 in fitslist1:
        cdsnum1 = re.match(r"spec\d{6}-\d{4}_CDS(\d{2})\.fits", os.path.basename(fits_path1)).group(1)
        cdsnum1_list.append(cdsnum1)
    for fits_path2 in fitslist2:
        cdsnum2 = re.match(r"spec\d{6}-\d{4}_CDS(\d{2})\.fits", os.path.basename(fits_path2)).group(1)
        cdsnum2_list.append(cdsnum2)
        
    # Convert to integers for numeric comparison
    nums1 = [int(x) for x in cdsnum1_list]
    nums2 = [int(x) for x in cdsnum2_list]

    # Find common CDS numbers
    common = sorted(set(nums1) & set(nums2))
    if len(common) < 2:
        return None

    cmin, cmax = common[0], common[-1]

    # Find indices in original lists
    idx1_min = nums1.index(cmin)
    idx2_min = nums2.index(cmin)
    idx1_max = nums1.index(cmax)
    idx2_max = nums2.index(cmax)

    return idx1_min, idx2_min, idx1_max, idx2_max


def create_CDS_image(fitspath1: str, fitspath2: str):
    cdsnum1 = re.match(r"spec\d{6}-\d{4}_CDS(\d{2})\.fits", os.path.basename(fitspath1)).group(1)
    cdsnum2 = re.match(r"spec\d{6}-\d{4}_CDS(\d{2})\.fits", os.path.basename(fitspath2)).group(1)
    new_fitspath = re.sub(r"CDS\d{2}\.fits", f"CDS{cdsnum1}-{cdsnum2}.fits", fitspath1)
    header_fitspath = re.sub(r"\.CDS\d{2}\.fits", ".CDS30.fits", fitspath1)
    fits1 = fits.getdata(fitspath1)
    fits2 = fits.getdata(fitspath2)
    header = fits.getheader(header_fitspath)
    image = fits1 - fits2
    fits.writeto(new_fitspath, image, header=header)
    return new_fitspath


def subtract_AB_image(fitspath1: str, fitspath2: str, savepath: str = WORK_DIR):
    imagenum1 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}-CDS(\d{2})\.fits", os.path.basename(fitspath1)).group(1)
    imagenum2 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}-CDS(\d{2})\.fits", os.path.basename(fitspath2)).group(1)
    new_fitspath = re.sub(r"\d{4}\_CDS\d{2}-CDS\d{2}\.fits", f"{imagenum1}-{imagenum2}.fits", fitspath1)
    fits1 = fits.getdata(fitspath1)
    fits2 = fits.getdata(fitspath2)
    header = fits.getheader(fitspath1)
    image = fits1 - fits2
    fits.writeto(new_fitspath, image, header=header)
    return new_fitspath



def reduction_main():
    if len(sys.argv) == 3:
        object_name = sys.argv[1]
        date_label = sys.argv[2]
    elif len(sys.argv) == 2:
        object_name = sys.argv[1]
        date_label = None
    conn = sqlite3.connect(DB_PATH)
    filepath_dict = db_search(conn, object_name, date_label)
    conn.close()
    for date_label, base_name_list in filepath_dict.items():
        do_scp_raw_fits(date_label, object_name, base_name_list)
        raw_fitslist = glob.glob(f"{RAWDATA_DIR}/{object_name}/{date_label}/*.fits")    
        raw_fitslist = quality_check(raw_fitslist)
        fitsdict = gen_fitsdict(raw_fitslist)
        label_dict = classify_spec_location(fitsdict)
        pair_label_list = search_combination_with_diff_label(label_dict)
        for no1, no2 in pair_label_list:
            fitslist1 = fitsdict[no1]
            fitslist2 = fitsdict[no2]
            fitslist1 = reject_saturation(fitslist1)
            fitslist2 = reject_saturation(fitslist2)
            fitslist1.sort()
            fitslist2.sort()
            idx1_min, idx2_min, idx1_max, idx2_max = search_combination_for_set_AB(fitslist1, fitslist2)
            cds1_path = create_CDS_image(fitslist1[idx1_max], fitslist2[idx1_min])
            cds2_path = create_CDS_image(fitslist1[idx2_max], fitslist2[idx2_min])
            subtract_AB_image(cds1_path, cds2_path)
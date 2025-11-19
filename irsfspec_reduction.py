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

from tools import spec_locator
from tools import classify_spec_location as csl

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

# Global log file paths for quality and saturation checks
QUALITY_LOG_PATH = None
SATURATION_LOG_PATH = None


# 出力ディレクトリを作成・返すヘルパー
def get_output_dir(object_name: str, date_label: str) -> Path:
    """WORK_DIR/object_name/date_label に対応する出力ディレクトリを返す。"""
    d = Path(WORK_DIR) / object_name / date_label
    d.mkdir(parents=True, exist_ok=True)
    return d



def _load_fits(fits_path: str) -> Tuple[fits.Header, np.ndarray]:
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS not found: {fits_path}")
    with fits.open(fits_path, memmap=False) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    if data is None:
        raise ValueError(f"No data in primary HDU: {fits_path}")
    data = np.asarray(data, dtype=np.float32)
    return header, data


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

    return matches[0]


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
    """品質チェックの結果をログにまとめて出力し、通過したファイルのみ返す。"""
    pass_list: List[str] = []
    fail_list: List[str] = []
    per_file_logs: List[str] = []

    for fits_path in fitslist:
        header, data = _load_fits(fits_path)
        area = compute_area(data)
        if area > 0.0:
            # 読み出しエラーとして扱う
            logging.warning(
                "quality_check: read error in %s (area=%f)",
                os.path.basename(fits_path),
                area,
            )
            fail_list.append(fits_path)
            per_file_logs.append(f"{os.path.basename(fits_path)}, NG, area={area:.6f}")
            continue
        pass_list.append(fits_path)
        per_file_logs.append(f"{os.path.basename(fits_path)}, OK, area={area:.6f}")

    # サマリーをログに出す
    logging.info(
        "quality_check summary: total=%d, pass=%d, fail=%d",
        len(fitslist),
        len(pass_list),
        len(fail_list),
    )
    if fail_list:
        logging.info(
            "quality_check failed files: %s",
            ", ".join(os.path.basename(p) for p in fail_list),
        )
    # Append summary to QUALITY_LOG_PATH if set
    if QUALITY_LOG_PATH is not None:
        with open(QUALITY_LOG_PATH, "a") as f:
            for line in per_file_logs:
                f.write(line + "\n")
            f.write(f"quality_check summary: total={len(fitslist)}, pass={len(pass_list)}, fail={len(fail_list)}\n")
    return pass_list


def gen_fitsdict(fitslist: List[str]) -> Dict[str, List[str]]:
    fitsdict: Dict[str, List[str]] = {}
    for fits_path in fitslist:
        num1 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}\.fits", os.path.basename(fits_path)).group(1)
        fitsdict.setdefault(num1, []).append(fits_path)
    # Return dict sorted by string num1 (insertion-ordered in Py3.7+)
    sorted_fitsdict = {k: fitsdict[k] for k in sorted(fitsdict.keys())}
    return sorted_fitsdict


'''
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
            darkpath = find_dark(fits_path, DARK4LOCATE_DIR)
            _, dark = _load_fits(darkpath)
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
'''

def classify_spec_location(fitsdict: Dict[str, List[str]]) -> Dict[str, str]:
    """
    fitsdict: {number: [fits_path, ...]}
    各 number について、mask != None となる最初のファイルの中心位置を使う。
    Returns: {No: label}
    """
    centers: Dict[str, float] = {}

    for number, fitslist in fitsdict.items():
        # CDS 番号の大きい順など、元の意図に合わせて並べ替え
        fitslist_sorted = sorted(fitslist, reverse=True)

        center_y = None

        for fits_path in fitslist_sorted:
            header, data = _load_fits(fits_path)
            darkpath = find_dark(fits_path, DARK4LOCATE_DIR)
            _, dark = _load_fits(darkpath)
            image = data - dark
            mask = spec_locator.spec_locator(image)
            if mask is None:
                # このファイルではスペクトルが見つからなかった → 次のファイルへ
                logging.warning(f"spec_locator failed for {os.path.basename(fits_path)}: skipping.")
                continue
            # 最初に見つかった mask を採用してループを抜ける
            center_y = csl.vertical_center_from_mask(mask)
            break

        if mask is None:
            # その番号の全ファイルで mask が見つからなかった → スキップ
            logging.warning(f"spec_locator failed for all files of No {number}; skipping.")
            continue
        
        centers[number] = center_y
        logging.info(f"No {number}: {center_y} in {os.path.basename(fits_path)}")

    if not centers:
        raise RuntimeError("No valid spectrum location found in any file.")

    # ここでようやく tools.classify_spec_location に渡す
    result = csl.classify_spec_location(centers)
    return result


def reject_saturation(fitslist: List[str]):
    """飽和チェックの結果をログにまとめて出力し、通過したファイルのみ返す。"""
    pass_fitslist: List[str] = []
    saturated_list: List[str] = []
    no_spec_list: List[str] = []

    for fits_path in fitslist:
        header, data = _load_fits(fits_path)
        mask = spec_locator.spec_locator(data)
        if mask is None:
            # スペクトル位置が特定できないフレームは使わない
            logging.warning(
                "reject_saturation: spec not found in %s, skipping.",
                os.path.basename(fits_path),
            )
            no_spec_list.append(fits_path)
            continue

        # スペクトル領域の画素値を取り出す
        spec_values = data[mask]

        # SATURATION_LEVEL 以下の値が 1 つでもあれば「飽和している」とみなして除外
        if np.any(spec_values <= SATURATION_LEVEL):
            logging.warning(
                "reject_saturation: saturated frame rejected: %s",
                os.path.basename(fits_path),
            )
            saturated_list.append(fits_path)
            continue

        # ここまで来たら飽和していないので採用
        pass_fitslist.append(fits_path)

    # サマリーをログに出す
    logging.info(
        "reject_saturation summary: total=%d, pass=%d, saturated=%d, no_spec=%d",
        len(fitslist),
        len(pass_fitslist),
        len(saturated_list),
        len(no_spec_list),
    )
    if saturated_list:
        logging.info(
            "reject_saturation saturated files: %s",
            ", ".join(os.path.basename(p) for p in saturated_list),
        )
    if no_spec_list:
        logging.info(
            "reject_saturation spec-not-found files: %s",
            ", ".join(os.path.basename(p) for p in no_spec_list),
        )
    # Append summary to SATURATION_LOG_PATH if set
    if SATURATION_LOG_PATH is not None:
        with open(SATURATION_LOG_PATH, "a") as f:
            f.write(f"reject_saturation summary: total={len(fitslist)}, pass={len(pass_fitslist)}, saturated={len(saturated_list)}, no_spec={len(no_spec_list)}\n")
            if saturated_list:
                f.write("saturated files: " + ", ".join(os.path.basename(p) for p in saturated_list) + "\n")
            if no_spec_list:
                f.write("spec-not-found files: " + ", ".join(os.path.basename(p) for p in no_spec_list) + "\n")

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


def create_CDS_image(fitspath1: str, fitspath2: str, savepath: Path):
    cdsnum1 = re.match(r"spec\d{6}-(\d{4})_CDS(\d{2})\.fits", Path(fitspath1).name).group(2)
    cdsnum2 = re.match(r"spec\d{6}-(\d{4})_CDS(\d{2})\.fits", Path(fitspath2).name).group(2)

    # 元のパスの basename だけを使って新しいファイル名を作る
    basename = Path(fitspath1).name
    new_basename = re.sub(r"CDS\d{2}\.fits", f"CDS{cdsnum1}-{cdsnum2}.fits", basename)
    new_fitspath = Path(savepath) / new_basename

    header, fits1 = _load_fits(fitspath1)
    _, fits2 = _load_fits(fitspath2)
    image = fits1 - fits2
    fits.writeto(new_fitspath, image, header=header, overwrite=True)
    return new_fitspath


def subtract_AB_image(fitspath1: str, fitspath2: str, savepath: Path):
    imagenum1 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}-\d{2}\.fits", Path(fitspath1).name).group(1)
    imagenum2 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}-\d{2}\.fits", Path(fitspath2).name).group(1)

    # 元のパスの basename から AB 減算後のファイル名を作る
    basename = Path(fitspath1).name
    new_basename = re.sub(r"\d{4}_CDS\d{2}-\d{2}\.fits", f"{imagenum1}-{imagenum2}.fits", basename)
    new_fitspath = Path(savepath) / new_basename

    header, fits1 = _load_fits(fitspath1)
    _, fits2 = _load_fits(fitspath2)
    image = fits1 - fits2
    header["IMAGE_A"] = Path(fitspath1).name
    header["IMAGE_B"] = Path(fitspath2).name
    fits.writeto(new_fitspath, image, header=header, overwrite=True)
    return new_fitspath



def reduction_main(object_name: str, date_label: str, base_name_list: List[str]):
    outdir = get_output_dir(object_name, date_label)

    # Set global log file paths for this reduction
    global QUALITY_LOG_PATH, SATURATION_LOG_PATH
    QUALITY_LOG_PATH = outdir / "quality_check.log"
    SATURATION_LOG_PATH = outdir / "reject_saturation.log"

    do_scp_raw_fits(date_label, object_name, base_name_list)
    raw_fitslist = glob.glob(f"{RAWDATA_DIR}/{object_name}/{date_label}/*.fits")
    raw_fitslist.sort()   
    raw_fitslist = quality_check(raw_fitslist)
    fitsdict = gen_fitsdict(raw_fitslist)
    label_dict = classify_spec_location(fitsdict)
    pair_label_list = search_combination_with_diff_label(label_dict)
    for no1, no2 in pair_label_list:
        fitslist1 = fitsdict[no1]
        fitslist2 = fitsdict[no2]

        # 飽和フレームを除外
        fitslist1 = reject_saturation(fitslist1)
        fitslist2 = reject_saturation(fitslist2)

        # 飽和除外の結果、どちらかが空になったらこのペアはスキップ
        if not fitslist1 or not fitslist2:
            logging.warning(f"No usable frames remain for pair (No {no1}, No {no2}) after saturation rejection; skipping.")
            continue

        fitslist1.sort()
        fitslist2.sort()

        combo = search_combination_for_set_AB(fitslist1, fitslist2)
        if combo is None:
            # 共通の CDS 番号が 2 つ未満 → AB 画像を作れないのでスキップ
            logging.warning(f"Not enough common CDS numbers between No {no1} and No {no2}; skipping.")
            continue

        idx1_min, idx2_min, idx1_max, idx2_max = combo

        cds1_path = create_CDS_image(fitslist1[idx1_min], fitslist1[idx1_max], outdir)
        cds2_path = create_CDS_image(fitslist2[idx2_min], fitslist2[idx2_max], outdir)
        subtract_AB_image(cds1_path, cds2_path, outdir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python irsfspec_reduction.py [object_name] [date_label]")
        sys.exit(1)
        
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
        reduction_main(object_name, date_label, base_name_list)
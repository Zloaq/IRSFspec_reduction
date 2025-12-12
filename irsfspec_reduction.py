#!/opt/anaconda3/envs/p11/bin/python3

from dotenv import load_dotenv
import os
import sys
import sqlite3
import subprocess
import logging
import logging.handlers
from pathlib import Path
import re
import glob
import astropy.io.fits as fits
import numpy as np
from itertools import combinations
from typing import Dict, List, Set, Tuple

from tools import spec_locator
from tools import classify_spec_location as csl

from astropy.utils.exceptions import AstropyWarning
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from collections import defaultdict
import tempfile

def setup_job_logger(outdir: Path, object_name: str, date_label: str) -> logging.Logger:
    """並列実行でもログが混ざらないよう、ジョブごとにファイルへ出力する。

    - outdir/reduction.log に書く（ジョブ単位で上書き）
    - 1 行の prefix に processName / object / date を入れる
    """
    logger = logging.getLogger(f"reduction.{object_name}.{date_label}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # root へ流さない（混線防止）

    # 既存ハンドラをクリア（二重出力防止）
    logger.handlers.clear()

    log_path = Path(outdir) / "reduction.log"
    fh = logging.FileHandler(log_path, mode="w")
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


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

NUM_PROCESS = 10


# 出力ディレクトリを作成・返すヘルパー
def get_output_dir(object_name: str, date_label: str) -> Path:
    """WORK_DIR/object_name/date_label に対応する出力ディレクトリを返す。"""
    d = Path(WORK_DIR) / object_name / date_label
    d.mkdir(parents=True, exist_ok=True)
    return d



def _load_fits(fits_path: Path) -> Tuple[fits.Header, np.ndarray]:
    fits_path = Path(fits_path)
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS not found: {fits_path}")
    with fits.open(str(fits_path), memmap=False) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    if data is None:
        raise ValueError(f"No data in primary HDU: {fits_path}")
    data = np.asarray(data, dtype=np.float32)
    return header, data


def find_dark(fitsname, darkpath) -> Path:
    fitsname = Path(fitsname)
    darkpath = Path(darkpath)
    fits_basename = fitsname.name
    m = re.search(r"(CDS\d{2})", fits_basename)
    if not m:
        raise ValueError(f"Could not extract CDS number from filename: {fits_basename}")
    cds_num = m.group(1)

    pattern = str(darkpath / f"*{cds_num}.fits")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No dark frame found for {cds_num} in {darkpath}")

    return Path(matches[0])




def db_search(conn: sqlite3.Connection, object_name, date_label=None) -> Dict[str, List[str]]:
    
    """
    framesテーブルからAr系のframeを抽出し、
    {date_label: [base_name, ...]} の辞書を返す。
    """
    if date_label is None:
        query = (
            "SELECT date_label, base_name "
            "FROM frames "
            f"WHERE filepath COLLATE NOCASE LIKE '%/spec%'" 
            f"AND object LIKE '{object_name}' "
        )
    else:
        query = (
            "SELECT date_label, base_name "
            "FROM frames "
            f"WHERE filepath COLLATE NOCASE LIKE '%/spec%'" 
            f"AND object LIKE '{object_name}' "
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
        src = f"{RAID_PC}:{RAID_DIR}/{date_label}/spec/spec{date_label}*{num1}*CDS*.fits*"
        dst = f"{RAWDATA_DIR}/{object_name}/{date_label}"
        os.makedirs(dst, exist_ok=True)
        cmd = ["scp", src, dst]

        # logging.info(f"COPY: {src}")
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except:
            logging.warning(
                "scp raised unexpected error (num1=%s) src=%s -> dst=%s. continuing.",
                num1,
                src,
                dst,
            )
            continue

import gzip
import shutil

def gunzip_if_needed(directory: Path, remove_gz: bool = True):
    """directory 内の *.fits.gz を解凍し、.fits を生成する。"""
    for gz_path in directory.glob("*.fits.gz"):
        fits_path = gz_path.with_suffix("")  # .gz を取る（*.fits になる）
        if fits_path.exists():
            # 既に解凍済みならスキップ
            continue

        with gzip.open(gz_path, "rb") as f_in, open(fits_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        if remove_gz:
            gz_path.unlink()
            #logging.info(f"Decompressed and removed: {gz_path.name} -> {fits_path.name}")
        else:
            #logging.info(f"Decompressed: {gz_path.name} -> {fits_path.name}")
            pass


def exptime_to_str(exptime_val) -> str:
    """EXP_TIME の値をファイル名用の文字列に変換する。

    - 整数なら 2 桁ゼロ埋め ("05", "10" など)
    - 少数なら小数点以下 3 桁までで末尾の 0 と小数点を削る
    - ファイル名に使えるように小数点は "p" に置換 (10.5 -> "10p5")
    """
    try:
        v = float(exptime_val)
    except Exception:
        # 数値にできなければそのまま str にする
        return str(exptime_val)

    if v.is_integer():
        return f"{int(v):02d}"

    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def do_average_noise(date_label: str):
    dst_dir = Path(RAWDATA_DIR) / "noise" / date_label
    if not dst_dir.exists():
        logging.warning(f"Noise directory does not exist: {dst_dir}")
        return
    
    noise_list = list(noise_dir.glob("spec*CDS*.fits"))
    pass_list, fail_list = quality_check(noise_list)
    if not pass_list:
        logging.warning(f"No noise files passed quality check in {dst_dir}")
        return


    noise_list = list(dst_dir.glob("spec*CDS30.fits"))
    if not noise_list:
        logging.warning(f"No noize files found in {dst_dir}")
        return
    logging.info("Noise averaging: found %d CDS30 frames in %s", len(noise_list), dst_dir)

    # EXP_TIME ごとにグループ分け
    groups = defaultdict(list)  # key: exptime (float), value: list[Path]
    for fits_path in noise_list:
        try:
            with warnings.catch_warnings():
                # Astropy の warning を例外に格上げして、壊れていそうな FITS を弾く
                warnings.simplefilter("error", AstropyWarning)
                with fits.open(fits_path, memmap=False) as hdul:
                    hdr = hdul[0].header
                    exptime = hdr.get("EXP_TIME", None)
            exptime_val = float(exptime)
        except AstropyWarning as e:
            logging.warning(f"AstropyWarning while reading header from {fits_path}: {e}. skipping.")
            continue
        except (TypeError, ValueError):
            logging.warning(f"failed to read EXP_TIME from {fits_path}. skipping.")
            continue
        except OSError as e:
            logging.warning(f"failed to open {fits_path}: {e}. skipping.")
            continue

        groups[exptime_val].append(fits_path)

    if not groups:
        logging.warning(f"No valid noise frames found in {dst_dir}")
        return
    logging.info("Noise averaging: %d EXP_TIME groups", len(groups))

    # 各 EXP_TIME ごとに、CDS01〜CDS30 を平均化
    for exptime_val, file_list in groups.items():
        if not file_list:
            continue

        exptime_str = exptime_to_str(exptime_val)
        logging.info(
            "Noise averaging: EXP_TIME=%s -> %d sequences (CDS01-30)",
            exptime_str,
            len(file_list),
        )
        created = 0
        skipped = 0

        # file_list 内には同じ EXP_TIME の specYYMMDD-NNNN_CDS30.fits が並んでいる想定
        for cds in range(1, 31):  # CDS01〜CDS30
            data_stack = []
            header_ref = None

            for fits_path in file_list:
                m = re.match(r"(spec\d{6}-\d{4})_CDS30\.fits$", fits_path.name)
                if not m:
                    logging.warning(f"Filename does not match pattern for CDS30: {fits_path.name}. skipping.")
                    continue

                prefix = m.group(1)  # specYYMMDD-NNNN
                target_name = f"{prefix}_CDS{cds:02d}.fits"
                target_path = fits_path.with_name(target_name)

                if not target_path.exists():
                    logging.warning(f"Missing CDS{cds:02d} file corresponding to {fits_path.name}: {target_path.name}. skipping.")
                    continue

                if target_path in fail_list:
                    logging.warning(f"File {target_path.name} failed quality check. skipping.")
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", AstropyWarning)
                        with fits.open(target_path, memmap=False) as hdul:
                            if header_ref is None:
                                # ヘッダは代表として 1 枚だけコピー
                                header_ref = hdul[0].header.copy()
                            data = hdul[0].data
                            if data is None:
                                logging.warning(f"No data in primary HDU: {target_path}")
                                continue
                            data_stack.append(np.asarray(data, dtype=np.float64))
                except AstropyWarning as e:
                    logging.warning(f"AstropyWarning while reading data from {target_path}: {e}. skipping.")
                    continue
                except OSError as e:
                    logging.warning(f"failed to open {target_path}: {e}. skipping.")
                    continue

            if not data_stack:
                logging.warning(
                    f"No valid frames to average for EXP_TIME={exptime_val}, CDS{cds:02d} in {dst_dir}. skipping."
                )
                skipped += 1
                # 進捗は 5 枚ごとに出す（行が増えすぎないように）
                if cds % 5 == 0:
                    logging.info(
                        "Noise averaging: EXP_TIME=%s progress %02d/30 (created=%d, skipped=%d)",
                        exptime_str,
                        cds,
                        created,
                        skipped,
                    )
                continue

            # 平均画像を作成
            stack = np.stack(data_stack, axis=0)
            mean_image = np.mean(stack, axis=0).astype(np.float32)

            if header_ref is None:
                header_ref = fits.Header()

            out_name = f"noise{date_label}_{exptime_str}_CDS{cds:02d}.fits"
            out_path = dst_dir / out_name

            fits.writeto(out_path, mean_image, header=header_ref, overwrite=True)
            created += 1
            if cds % 5 == 0:
                logging.info(
                    "Noise averaging: EXP_TIME=%s progress %02d/30 (created=%d, skipped=%d)",
                    exptime_str,
                    cds,
                    created,
                    skipped,
                )

        logging.info(
            "Noise averaging: EXP_TIME=%s done (created=%d, skipped=%d)",
            exptime_str,
            created,
            skipped,
        )


def load_noise(date_label: str, exptime, cds: int | None = None):
    """平均化済みノイズフレームを読み込むヘルパー。

    date_label: 観測日付（YYMMDD など）
    exptime: EXP_TIME / EXPTIME の値
    cds: 読み出したい CDS 番号（None の場合は 30 をデフォルトとする）
    """
    exptime_str = exptime_to_str(exptime)
    if cds is None:
        cds = 30

    noise_path = (
        Path(RAWDATA_DIR)
        / "noise"
        / date_label
        / f"noise{date_label}_{exptime_str}_CDS{cds:02d}.fits"
    )
    if not noise_path.exists():
        logging.warning(f"Noise file does not exist: {noise_path}")
        return None
    with fits.open(noise_path, memmap=False) as hdul:
        header = hdul[0].header
        data = hdul[0].data.astype(np.float64)

    return header, data


def load_dark(fitspath):
    """指定されたフレームに対応する平均ノイズ（dark）を読み込む。

    - EXP_TIME は対応する CDS30 フレームから取得
    - ノイズファイルは noise{date_label}_{exptime_str}_CDSxx.fits という命名に合わせる
    """
    fitspath = Path(fitspath)
    if not fitspath.exists():
        raise FileNotFoundError(f"FITS not found: {fitspath}")

    # 元ファイル名から CDS 番号を取得
    m = re.search(r"CDS(\d{2})\.fits$", fitspath.name)
    if not m:
        raise ValueError(f"Could not extract CDS number from filename: {fitspath.name}")
    cds_num = int(m.group(1))

    # EXP_TIME は同じ Num1 の CDS30 から取得する
    cds30_name = re.sub(r"CDS\d{2}\.fits", "CDS30.fits", fitspath.name)
    cds30_fitspath = fitspath.with_name(cds30_name)

    date_label = fitspath.parent.name
    header, _ = _load_fits(cds30_fitspath)
    exptime = header["EXPTIME"]
    exptime_str = exptime_to_str(exptime)

    dst_dir = Path(RAWDATA_DIR) / "noise" / date_label
    noise_path = dst_dir / f"noise{date_label}_{exptime_str}_CDS{cds_num:02d}.fits"
    if not noise_path.exists():
        logging.warning(f"Noise file does not exist: {noise_path}")
        return None

    with fits.open(noise_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float64)

    return data


def do_remove_raw_fits(date_label: str, object_name: str):
    target_dir = Path(RAWDATA_DIR) / object_name / date_label
    cmd = ["rm", "-f", str(target_dir / "spec*-????_CDS*.fits")]
    subprocess.run(" ".join(cmd), shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)




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


def quality_check(fitslist: List[Path]):
    """品質チェックの結果をログにまとめて出力し、通過したファイルのみ返す。"""
    pass_list: List[Path] = []
    fail_list: List[Path] = []
    per_file_logs: List[str] = []

    for fits_path in fitslist:
        header, data = _load_fits(fits_path)
        area = compute_area(data)
        if area > 0.0:
            # 読み出しエラーとして扱う
            logging.warning(
                "quality_check: read error in %s (area=%f)",
                fits_path.name,
                area,
            )
            fail_list.append(fits_path)
            # ログファイルは CSV 風の 3 カラム: filename,status,area
            per_file_logs.append(f"{fits_path.name},NG,{area:.6f}")
            continue
        pass_list.append(fits_path)
        per_file_logs.append(f"{fits_path.name},OK,{area:.6f}")


    # QUALITY_LOG_PATH に CSV 風で出力
    if QUALITY_LOG_PATH is not None:
        with open(QUALITY_LOG_PATH, "w") as f:
            # ヘッダ行
            f.write("filename,status,area\n")
            for line in per_file_logs:
                f.write(line + "\n")
            # サマリ行（コメント扱い）
            f.write(
                f"# summary: total={len(fitslist)}, pass={len(pass_list)}, fail={len(fail_list)}\n"
            )
    return pass_list, fail_list


def gen_fitsdict(fitslist: List[Path]) -> Dict[str, List[Path]]:
    fitsdict: Dict[str, List[Path]] = {}
    for fits_path in fitslist:
        num1 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}\.fits", fits_path.name).group(1)
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

def classify_spec_location(fitsdict: Dict[str, List[Path]]) -> Dict[str, str]:
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
        mask = None

        for fits_path in fitslist_sorted:

            header, data = _load_fits(fits_path)
            dark = load_dark(fits_path)
            if dark is None:
                darkpath = find_dark(fits_path, DARK4LOCATE_DIR)
                _, dark = _load_fits(darkpath)


            image = data - dark
            mask = spec_locator.spec_locator(image)
            if mask is None:
                # このファイルではスペクトルが見つからなかった → 次のファイルへ
                continue
            # 最初に見つかった mask を採用してループを抜ける
            center_y = csl.vertical_center_from_mask(mask)
            break

        if mask is None:
            # その番号の全ファイルで mask が見つからなかった → スキップ
            logging.warning(f"spec_locator failed for all files of No {number}; skipping.")
            continue

        centers[number] = center_y

    if not centers:
        return None

    # ここでようやく tools.classify_spec_location に渡す
    result = csl.classify_spec_location(centers)
    return result


def reject_saturation(fitslist: List[Path]):
    """飽和チェックの結果をログにまとめて出力し、通過したファイルのみ返す。"""
    pass_fitslist: List[Path] = []
    saturated_list: List[Path] = []
    no_spec_list: List[Path] = []
    per_file_logs: List[str] = []

    for fits_path in fitslist:
        header, data = _load_fits(fits_path)
        mask = spec_locator.spec_locator(data)
        if mask is None:
            # スペクトル位置が特定できないフレームもゆるそう
            no_spec_list.append(fits_path)
            pass_fitslist.append(fits_path)
            # filename,status
            per_file_logs.append(f"{fits_path.name},NO_SPEC")
            continue

        # スペクトル領域の画素値を取り出す
        spec_values = data[mask]

        # SATURATION_LEVEL 以下の値が 1 つでもあれば「飽和している」とみなして除外
        if np.any(spec_values <= SATURATION_LEVEL):
            saturated_list.append(fits_path)
            per_file_logs.append(f"{fits_path.name},SATURATED")
            continue

        # spec があって、飽和していないので採用
        pass_fitslist.append(fits_path)
        per_file_logs.append(f"{fits_path.name},OK")

    # SATURATION_LOG_PATH に CSV 風で出力
    if SATURATION_LOG_PATH is not None:
        with open(SATURATION_LOG_PATH, "w") as f:
            # ヘッダ行
            f.write("filename,status\n")
            for line in per_file_logs:
                f.write(line + "\n")
            # サマリ行（コメント扱い）
            f.write(
                f"# summary: total={len(fitslist)}, "
                f"pass={len(pass_fitslist)}, saturated={len(saturated_list)}, "
                f"no_spec={len(no_spec_list)}\n"
            )

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



def search_combination_for_set_AB(fitslist1: List[Path], fitslist2: List[Path]):
    results: List[Tuple[str, str]] = []
    cdsnum1_list = []
    cdsnum2_list = []
    for fits_path1 in fitslist1:
        cdsnum1 = re.match(r"spec\d{6}-\d{4}_CDS(\d{2})\.fits", fits_path1.name).group(1)
        cdsnum1_list.append(cdsnum1)
    for fits_path2 in fitslist2:
        cdsnum2 = re.match(r"spec\d{6}-\d{4}_CDS(\d{2})\.fits", fits_path2.name).group(1)
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


def create_CDS_image(fitspath1: Path, fitspath2: Path, savepath: Path) -> Path:
    fitspath1 = Path(fitspath1)
    fitspath2 = Path(fitspath2)
    savepath = Path(savepath)

    cdsnum1 = re.match(r"spec\d{6}-(\d{4})_CDS(\d{2})\.fits", fitspath1.name).group(2)
    cdsnum2 = re.match(r"spec\d{6}-(\d{4})_CDS(\d{2})\.fits", fitspath2.name).group(2)

    # 元のパスの basename だけを使って新しいファイル名を作る
    basename = fitspath1.name
    new_basename = re.sub(r"CDS\d{2}\.fits", f"CDS{cdsnum1}-{cdsnum2}.fits", basename)
    new_fitspath = savepath / new_basename

    header, fits1 = _load_fits(fitspath1)
    _, fits2 = _load_fits(fitspath2)

    cds30_name = re.sub(r"CDS\d{2}\.fits", "CDS30.fits", fitspath1.name)
    cds30_fitspath = fitspath1.with_name(cds30_name)
    if cds30_fitspath.exists():
        header, _ = _load_fits(cds30_fitspath)

    image = fits1 - fits2
    fits.writeto(new_fitspath, image, header=header, overwrite=True)
    return new_fitspath


def subtract_AB_image(fitspath1: Path, fitspath2: Path, savepath: Path) -> Path:
    fitspath1 = Path(fitspath1)
    fitspath2 = Path(fitspath2)
    savepath = Path(savepath)

    imagenum1 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}-\d{2}\.fits", fitspath1.name).group(1)
    imagenum2 = re.match(r"spec\d{6}-(\d{4})_CDS\d{2}-\d{2}\.fits", fitspath2.name).group(1)

    # 元のパスの basename から AB 減算後のファイル名を作る
    basename = fitspath1.name
    new_basename = re.sub(r"\d{4}_CDS\d{2}-\d{2}\.fits", f"{imagenum1}-{imagenum2}.fits", basename)
    new_fitspath = savepath / new_basename

    header, fits1 = _load_fits(fitspath1)
    _, fits2 = _load_fits(fitspath2)

    cds30_name1 = re.sub(r"CDS\d{2}-\d{2}\.fits", "CDS30.fits", fitspath1.name)
    cds30_name2 = re.sub(r"CDS\d{2}-\d{2}\.fits", "CDS30.fits", fitspath2.name)
    cds30_fitspath1 = fitspath1.with_name(cds30_name1)
    cds30_fitspath2 = fitspath2.with_name(cds30_name2)
    if cds30_fitspath1.exists():
        header, _ = _load_fits(cds30_fitspath1)
    elif cds30_fitspath2.exists():
        header, _ = _load_fits(cds30_fitspath2)

    image = fits1 - fits2
    header["IMAGE_A"] = fitspath1.name
    header["IMAGE_B"] = fitspath2.name
    fits.writeto(new_fitspath, image, header=header, overwrite=True)
    return new_fitspath



def reduction_main(object_name: str, date_label: str, base_name_list: List[str]):
    outdir = get_output_dir(object_name, date_label)

    # Set up per-job logger
    logger = setup_job_logger(outdir, object_name, date_label)

    # Set global log file paths for this reduction
    global QUALITY_LOG_PATH, SATURATION_LOG_PATH
    QUALITY_LOG_PATH = outdir / "quality_check.log"
    SATURATION_LOG_PATH = outdir / "reject_saturation.log"

    print()
    logger.info("==== Start reduction: object=%s, date_label=%s ====", object_name, date_label)
    logger.info("Output directory: %s", outdir)

    do_scp_raw_fits(date_label, object_name, base_name_list)
    datadir = Path(RAWDATA_DIR) / object_name / date_label
    gunzip_if_needed(datadir)

    raw_fitslist = sorted(datadir.glob("*.fits"))

    logger.info("Found %d raw FITS files before quality check.", len(raw_fitslist))

    n_before_quality = len(raw_fitslist)
    raw_fitslist, _ = quality_check(raw_fitslist)
    logger.info(
        "Quality check result: %d -> %d files",
        n_before_quality,
        len(raw_fitslist),
    )
    if not raw_fitslist:
        logger.warning(
            "No usable raw FITS files for object=%s, date_label=%s after quality check; skipping.",
            object_name,
            date_label,
        )
        return

    fitsdict = gen_fitsdict(raw_fitslist)
    logger.info("Number of Num1 groups after quality check: %d", len(fitsdict))

    label_dict = classify_spec_location(fitsdict)
    if label_dict is None:
        logger.warning(
            "No usable raw FITS files for object=%s, date_label=%s after classification; skipping.",
            object_name,
            date_label,
        )
        return
    logger.info("Classified spectrum locations for %d groups.", len(label_dict))
    
    if len(label_dict) == 1:
        logger.warning(
            "Only one group for object=%s, date_label=%s; skipping.",
            object_name,
            date_label,
        )
        return

    pair_label_list = search_combination_with_diff_label(label_dict)
    logger.info("Found %d AB pairs with different labels.", len(pair_label_list))

    for no1, no2 in pair_label_list:
        logger.info("== Processing pair: No %s (group1) and No %s (group2)", no1, no2)
        fitslist1 = fitsdict[no1]
        fitslist2 = fitsdict[no2]

        # 飽和フレームを除外
        fitslist1 = reject_saturation(fitslist1)
        fitslist2 = reject_saturation(fitslist2)

        logger.info(
            "After saturation rejection: No %s -> %d frames, No %s -> %d frames",
            no1,
            len(fitslist1),
            no2,
            len(fitslist2),
        )

        # 飽和除外の結果、どちらかが空になったらこのペアはスキップ
        if not fitslist1 or not fitslist2:
            logger.warning(
                "No usable frames remain for pair (No %s, No %s) after saturation rejection; skipping.",
                no1,
                no2,
            )
            continue

        fitslist1.sort()
        fitslist2.sort()

        combo = search_combination_for_set_AB(fitslist1, fitslist2)
        if combo is None:
            # 共通の CDS 番号が 2 つ未満 → AB 画像を作れないのでスキップ
            logger.warning(
                "Not enough common CDS numbers between No %s and No %s; skipping.",
                no1,
                no2,
            )
            continue

        idx1_min, idx2_min, idx1_max, idx2_max = combo
        logger.info(
            "Using CDS indices: No %s -> (%d, %d), No %s -> (%d, %d)",
            no1,
            idx1_min + 1,
            idx1_max + 1,
            no2,
            idx2_min + 1,
            idx2_max + 1,
        )

        cds1_path = create_CDS_image(fitslist1[idx1_min], fitslist1[idx1_max], outdir)
        cds2_path = create_CDS_image(fitslist2[idx2_min], fitslist2[idx2_max], outdir)

        ab_path = subtract_AB_image(cds1_path, cds2_path, outdir)
        logger.info("Created AB-subtracted image: %s", ab_path.name)

    logger.info("==== End reduction: object=%s, date_label=%s ====", object_name, date_label)


def worker_init() -> None:
    """ProcessPoolExecutor 用の初期化関数。ワーカープロセスごとにロガーをクリアする。"""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # ワーカープロセスは root へは出さない（混線防止）
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(logging.NullHandler())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(processName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
    darkpath_dict = db_search(conn, "dark", date_label)
    conn.close()

    for date_label, base_name_list in darkpath_dict.items():
        noise_dir = Path(f"{RAWDATA_DIR}/noise/{date_label}")
        if not noise_dir.exists():
            do_scp_raw_fits(date_label, "noise", base_name_list)
            gunzip_if_needed(noise_dir, remove_gz=True)
            do_average_noise(date_label)
            do_remove_raw_fits(date_label, "noise")
    
    with ProcessPoolExecutor(max_workers=NUM_PROCESS, initializer=worker_init) as ex:
        future_to_job = {
            ex.submit(reduction_main, object_name, date_label, base_name_list): (object_name, date_label, base_name_list)
            for date_label, base_name_list in filepath_dict.items()
        }
        total = len(future_to_job)
        done = 0
        for future in as_completed(future_to_job):
            object_name, date_label, base_name_list = future_to_job[future]
            try:
                future.result()
                status = "finished"
            except Exception as e:
                logging.exception(
                    "Job failed: object=%s date_label=%s (continuing)",
                    object_name,
                    date_label,
                )
                status = "failed"

            done += 1
            sys.stdout.write(f"[{done}/{total}] {status} {object_name} {date_label}\n")
            sys.stdout.flush()

    # for date_label, base_name_list in filepath_dict.items():
    #     reduction_main(object_name, date_label, base_name_list)
    #     do_remove_raw_fits(date_label, object_name)
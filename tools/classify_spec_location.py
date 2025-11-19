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

    #print(centerdict)
    #print(result)
    return result



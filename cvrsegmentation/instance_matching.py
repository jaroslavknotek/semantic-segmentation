import numpy as np
import torch
import itertools
import cv2
from dataclasses import dataclass
import pandas as pd
import numpy.typing as npt
from torchmetrics.classification import BinaryJaccardIndex

import scipy.optimize

import logging


logger = logging.getLogger("cvrsegmentation.instance_matching")


@dataclass
class SegmentationInstance:
    """Instance Shape"""

    mask: npt.NDArray
    image: npt.NDArray
    top_left_x: int
    top_left_y: int
    width: int
    height: int


@dataclass
class SegmentationInstanceFeatures:
    """Precipitate features"""

    ellipse_width_px: float
    ellipse_height_px: float
    ellipse_center_x: float
    ellipse_center_y: float
    ellipse_angle_deg: float
    circle_x: float
    circle_y: float
    circle_radius: float
    area_px: float
    shape: str


def measure_precision_recall_f1(bin_prediction, label, component_limit=500):
    df, _, _ = match_instances(
        bin_prediction,
        label,
        component_limit=component_limit,
    )

    grains_pred = len(df[~df["pred_id"].isna()])
    grains_label = len(df[~df["label_id"].isna()])

    tp = df[~df["label_id"].isna() & ~df["pred_id"].isna()]
    if grains_pred != 0:
        precision = len(tp) / grains_pred
    else:
        precision = np.nan

    if grains_label != 0:
        recall = len(tp) / grains_label
    else:
        recall = np.nan

    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def match_instances(bin_prediction, label, component_limit=500):
    p_n, p_grains = cv2.connectedComponents(bin_prediction)
    l_n, l_grains = cv2.connectedComponents(label)

    if p_n > component_limit or l_n > component_limit:
        logger.warning(
            f"Too many components found {component_limit=} #predictions:{p_n} #labels:{l_n}. Cropping"
        )
        p_n = min(p_n, component_limit)
        p_grains[p_grains > component_limit] = 0
        l_n = min(l_n, component_limit)
        l_grains[l_grains > component_limit] = 0

    # pairs only #TP
    pred_items, label_items = _pair_using_linear_sum_assignment(
        p_n, p_grains, l_n, l_grains
    )
    data = list(zip(pred_items, label_items))

    # FP
    p_set = set(pred_items)
    false_positives = [i for i in range(1, p_n) if i not in p_set]
    for i in false_positives:
        data.append((i, None))

    # FN
    l_set = set(label_items)
    label_positives = [i for i in range(1, l_n) if i not in l_set]
    for i in label_positives:
        data.append((None, i))
    df = pd.DataFrame(data, columns=["pred_id", "label_id"])
    return df, p_grains, l_grains


def prepare_iou(foreground_thr=0.5):
    m = BinaryJaccardIndex(threshold=foreground_thr)

    def met(a, b):
        a = np.where(a < foreground_thr, 0, 1)
        b = np.where(b < foreground_thr, 0, 1)
        return m(torch.Tensor(a), torch.Tensor(b))

    return met


def _pair_using_linear_sum_assignment(p_n, p_grains, l_n, l_grains, cap=500):
    if cap is not None:
        p_n = min(cap, p_n)
        p_grains[p_grains > cap] = 0

        l_n = min(cap, l_n)
        l_grains[l_grains > cap] = 0

    weights_dict = _collect_pairing_weights(p_n, p_grains, l_n, l_grains)
    weights, p_map, l_map = _construct_weight_map(weights_dict)
    p_item_id, l_item_id = scipy.optimize.linear_sum_assignment(weights)

    inverse_p_map = {v: k for k, v in p_map.items()}
    p_item = np.array([inverse_p_map[idx] for idx in p_item_id])
    inverse_l_map = {v: k for k, v in l_map.items()}
    l_item = np.array([inverse_l_map[idx] for idx in l_item_id])
    return p_item, l_item


def _construct_weight_map(weights_dict):
    p_map = {}

    for i, v in enumerate(weights_dict.keys()):
        p_map[v] = i

    l_keys = itertools.chain(
        *(list(k for k in v.keys()) for v in weights_dict.values())
    )
    l_unique = np.unique(list(l_keys))
    l_map = {}
    for i, v in enumerate(l_unique):
        l_map[v] = i

    weights = np.zeros((len(p_map), len(l_map)))
    for i, (p, pv) in enumerate(weights_dict.items()):
        for ll, lv in pv.items():
            weights[p_map[p], l_map[ll]] = lv
    return weights, p_map, l_map


def _collect_pairing_weights(p_n, p_grains, l_n, l_grains):
    weights_dict = {}
    iou = prepare_iou()
    for p_grain_id in range(1, p_n):
        p_grain_mask = np.uint8(p_grains == p_grain_id)

        intersecting_ids = np.unique(l_grains * p_grain_mask)
        intersecting_ids = intersecting_ids[intersecting_ids > 0]

        for l_grain_id in intersecting_ids:
            l_grain_mask = np.uint8(l_grains == l_grain_id)

            weight = 1 - iou(l_grain_mask, p_grain_mask)
            weights_dict.setdefault(p_grain_id, {}).setdefault(l_grain_id, weight)

    return weights_dict

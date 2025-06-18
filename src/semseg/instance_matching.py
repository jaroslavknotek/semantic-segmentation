import numpy as np
import torch
import itertools
import pandas as pd
import cv2
from dataclasses import dataclass
import pandas as pd
import numpy.typing as npt
from torchmetrics.classification import BinaryJaccardIndex
from tqdm.cli import tqdm
import scipy.optimize
import semseg.imageutils as iu
import semseg.feret as fer
import logging

logger = logging.getLogger("semseg.instance_matching")


@dataclass
class SegmentationInstanceFeatures:
    """Precipitate features"""
    
    area_px:float
    area_per:float
    d0:float
    roundness:float
    feret_min_px:float
    feret_max_px:float
    feret_90_px:float
    slope:float
   

@dataclass
class BoundingBox:
    top:int
    left:int
    bottom:int
    right:int

    @property
    def width(self):
        return self.right - self.left
    
    @property
    def height(self):
        return self.bottom - self.top
    
@dataclass
class SegmentationInstance:
    """Instance Shape"""
    mask: npt.NDArray
    image: npt.NDArray
    bounding_box: BoundingBox



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
    
    pr = precision + recall
    if pr == 0:
        return np.nan,np.nan,np.nan
    
    f1 = 2 * (precision * recall) / pr
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


def measure_labeled(
    thr, 
    labels,
    predictions,
    small_filter,
    component_limit=1000,
    use_tqdm = False
):
    precisions = []
    recalls = []
    for label,prediction in tqdm(list(zip(labels,predictions)),disable=not use_tqdm):
        pred_thr = np.uint8(prediction>=thr)
        label_thr = np.uint8(label)
        pred_thr = iu.filter_small(pred_thr,small_filter)
        
        precision,recall,f1 = measure_precision_recall_f1(pred_thr,label_thr,component_limit=component_limit)
        precisions.append(precision)
        recalls.append(recall)
        
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = 2* (mean_precision*mean_recall)/(mean_precision + mean_recall)
    return mean_precision,mean_recall, mean_f1



def extract_individuals(img,mask):
    n,arr = cv2.connectedComponents(mask)
    
    masks = [np.uint8(arr == i) for i in range(1,n)]
    bbs = list(map(get_bounding_box,masks))
    
    cropped_masks = (mask[bb.top:bb.bottom,bb.left:bb.right] for bb in bbs)
    cropped_imgs = (img[bb.top:bb.bottom,bb.left:bb.right] for bb in bbs)
    return [
        SegmentationInstance(
            mask = mask_cropped,
            image = img_cropped,
            bounding_box = bb
        )
        for img_cropped,mask_cropped,bb
        in zip(cropped_imgs,cropped_masks,bbs)
    ]
    
    #areas = np.array([np.sum(small_mask) for small_mask in masks])
    # if include_mask:
    #     masks_cropped = map(_crop_to_content,masks)
    #     data = list(zip(areas,masks_cropped))
    #     return pd.DataFrame(data,columns=['area_px','mask'])
    # else:
    #     return pd.DataFrame(areas[None].T,columns=['area_px'])

def cleanup_prediction(pred, thr, small_filter):
    filtered_pred = iu.filter_small(pred>thr,small_filter)
    return np.uint8(filtered_pred)


def get_bounding_box(mask):
    nz = np.nonzero(mask)
    t,b = np.min(nz[0]), np.max(nz[0])
    l,r = np.min(nz[1]), np.max(nz[1])
    return BoundingBox(top=t,left=l,bottom=b,right = r)

def calculate_features(instance, full_mask):
    prec_mask = instance.mask
    bb = instance.bounding_box
    area_px = int(np.sum(prec_mask))
    
    
    feret_mask = np.pad(prec_mask,1)
    (feret_min,feret_max),(_,max_points) = fer.get_min_max_feret_from_mask(feret_mask)
    feret_90, slope = fer.get_feret90_w_slope(feret_mask, max_points)

    return SegmentationInstanceFeatures(
        area_px = np.sum(prec_mask),
        area_per = float(area_px/np.prod(full_mask.shape)),
        d0 = np.sqrt(area_px/np.pi),
        roundness = np.min([bb.height,bb.width])/np.max([bb.height,bb.width]),
        feret_min_px = feret_min,
        feret_max_px = feret_max,
        feret_90_px = feret_90,
        slope = slope,
    )
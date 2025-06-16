import cv2
import numpy as np


def extract_components_sorted(bin_img, connectivity=8):
    n, arr = cv2.connectedComponents(bin_img, connectivity=connectivity)

    areas = np.array([np.uint8(arr == i) for i in range(1, n)])
    area_sizes = [np.sum(area) for area in areas]
    arg_sorted = np.argsort(area_sizes)
    return areas[arg_sorted]

def fill_holes(bin_mask):
    des = np.zeros_like(bin_mask)
    contours,hier = cv2.findContours(bin_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(des,contours,0,1,-1)
    return des


def filter_small(mask,area_limit:int):
    if area_limit == 0:
        return mask
    
    old_dtype = mask.dtype
    mask = np.uint8(mask)
    n,lbs = cv2.connectedComponents(mask)
    base = np.zeros_like(mask,dtype=np.uint8)
    for i in range(1,n):
        component = np.uint8(lbs==i)
        size = np.sum(component)
        if size >= area_limit:
            base = base | component
    return base.astype(old_dtype)

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

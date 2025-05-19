import semseg.feret
import skimage.morphology
import scipy.ndimage as ndi
import numpy as np
import skimage.morphology
import semseg.instance_matching as im
import semseg.feret as fer
#from rotating_calipers import min_max_feret

# Taken from https://gist.github.com/VolkerH/0d07d05d5cb189b56362e8ee41882abf
def get_min_max_feret_from_labelim(label_im, labels=None):
    """ given a label image, calculate the oriented 
    bounding box of each connected component with 
    label in labels. If labels is None, all labels > 0
    will be analyzed.
    Parameters:
        label_im: numpy array with labelled connected components (integer)
    Output:
        obbs: dictionary of oriented bounding boxes. The dictionary 
        keys correspond to the respective labels
    """
    if labels is None:
        labels = set(np.unique(label_im)) - {0}
    results = {}
    for label in labels:
        results[label] = get_min_max_feret_from_mask(label_im == label)
    return results

def get_min_max_feret_from_mask(mask_im):
    """ given a binary mask, calculate the minimum and maximum
    feret diameter of the foreground object. This is done
    by calculating the outline of the object, transform
    the pixel coordinates of the outline into a list of
    points and then calling 
    Parameters:
        mask_im: binary numpy array
    """
    eroded = skimage.morphology.erosion(mask_im)
    outline = mask_im ^ eroded
    boundary_points = np.argwhere(outline > 0)
    # convert numpy array to a list of (x,y) tuple points
    boundary_point_list = list(map(list, list(boundary_points)))
    return min_max_feret(boundary_point_list)




# This file contains code taken from 
# http://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/
# convex hull (Graham scan by x-coordinate) and diameter of a set of points
# David Eppstein, UC Irvine, 7 Mar 2002

# According to that website the code is under the PSF licencse
# https://en.wikipedia.org/wiki/Python_Software_Foundation_License


# modifications by Volker Hilsenstein
from math import sqrt


def orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def hulls(Points):
    '''Graham scan to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def rotatingCalipers(Points):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    U,L = hulls(Points)
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j]
        
        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1
        
        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1


def min_max_feret(Points):
    '''Given a list of 2d points, returns the minimum and maximum feret diameters.'''
    squared_distance_per_pair = [
        ((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
        for p,q in rotatingCalipers(Points)
    ]
    min_feret_sq, min_feret_pair = min(squared_distance_per_pair)
    max_feret_sq, max_feret_pair = max(squared_distance_per_pair)
    return sqrt(min_feret_sq), sqrt(max_feret_sq)


def diameter(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    diam,pair = max([
        ((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
        for p,q in rotatingCalipers(Points)
    ])
    return diam, pair

def min_feret(Points):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    min_feret_sq,pair = min([
        ((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
        for p,q in rotatingCalipers(Points)
    ])
    return min_feret_sq, pair




def get_min_max_feret_from_mask(mask_im):
    """ given a binary mask, calculate the minimum and maximum
    feret diameter of the foreground object. This is done
    by calculating the outline of the object, transform
    the pixel coordinates of the outline into a list of
    points and then calling 
    Parameters:
        mask_im: binary numpy array
    """
    eroded = skimage.morphology.erosion(mask_im)
    outline = mask_im ^ eroded
    boundary_points = np.argwhere(outline > 0)
    # convert numpy array to a list of (x,y) tuple points
    boundary_point_list = list(map(list, list(boundary_points)))
    return min_max_feret(boundary_point_list)

def min_max_feret(Points):
    '''Given a list of 2d points, returns the minimum and maximum feret diameters.'''
    squared_distance_per_pair = [
        ((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
        for p,q in rotatingCalipers(Points)
    ]
    min_feret_sq, min_feret_pair = min(squared_distance_per_pair)
    max_feret_sq, max_feret_pair = max(squared_distance_per_pair)
    
    
    ferets = np.sqrt([min_feret_sq,max_feret_sq])
    points = min_feret_pair, max_feret_pair
    return ferets, np.array(points)

def get_feret90_w_slope(mask, max_ferret):
    (y1,x1),(y2,x2) = max_ferret
    x = x1-x2
    y = y1-y2 

    if x == 0:
        slope = np.nan
        feret90 = np.nan
    else:
        slope = y/x
        rad = np.arctan([y/x])
        deg=float(np.rad2deg(rad))
        rotated_mask = ndi.rotate(mask,deg,reshape=True)
        bb = im.get_bounding_box(rotated_mask)
        feret90 = bb.bottom - bb.top

    

    
    
    
    return feret90, slope
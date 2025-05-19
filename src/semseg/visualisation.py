import matplotlib.pyplot as plt

import cv2
import numpy as np

def plot_particles(img,individuals,title):
    fig,ax = plt.subplots(1,1,figsize = (12,12))
    ax.set_title(title)
    ax.imshow(img,cmap='gray')

    for i,ind in enumerate(individuals):

        contours_arr, _ = cv2.findContours(
            ind.mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        contour = contours_arr[0]
        ctr = np.concatenate([contour,[contour[0]]])

        t = ind.bounding_box.top
        l = ind.bounding_box.left

        #print(ctr,l,t)
        ctr += [l,t]
        
        ax.plot(*ctr.T,c='r')
        ax.text(l,t,i,c='green',weight = "bold", 
            bbox=dict(boxstyle="square",
               ec=(0,0,0,0),
               fc=(1, 1, 1,.2),
            )
        )
    
    return fig
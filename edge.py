# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:05:41 2017

@author: nc17
"""

import rasterio
import numpy as np
from skimage.filters import roberts, sobel, scharr, prewitt


filename = "../training_data/training_new_IMG_3451.tif"

def main():
    with rasterio.open(filename) as src:
        image = src.read(masked=True)
        mask = src.read_masks(1)
        
        output = np.zeros(tuple([4]) + src.shape)
        
        output[0] = roberts(image[0], mask)
        output[1] = sobel(image[0], mask)
        output[2] = scharr(image[0], mask)
        output[3] = prewitt(image[0], mask)
        
        r_meta = src.meta
        r_meta['dtype'] = 'float64'
        r_meta['count'] = 4
        
        with rasterio.open("sobel.tif", "w", **r_meta) as dst:
            dst.write(output)

if __name__ == "__main__":
    main()
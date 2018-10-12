#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:50:07 2018

@author: nate
"""
from . import utility
import numpy as np
from collections import OrderedDict
from rasterstats import zonal_stats
from skimage.segmentation import slic
from skimage.measure import regionprops, label
from rasterio import features


def is_working():
    print("train.py is working!")


def slic_segmentation(image, mask):
    """
    Segment the image.

    Segment the image using the slic algorithm (from sklearn.segmentation).

    Parameters
    ----------
    image: numpy.array
        A rasterio-style image. Obtained and transformed by:
        src.read(masked=True).transpose(1, 2, 0)

    mask: numpy.array
        A rasterio-style mask. Obtained by src.read_masks(1)
        This function doesn't do anything with mask at the moment.
        This function assumes image has, and is read with, a mask.

    Returns
    -------
    numpy.array
        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
        so it's ready to be written by rasterio

    """
    segs = slic(image, n_segments=5000, compactness=1, slic_zero=True)
    mask[mask == 255] = 1
    output = segs * mask[np.newaxis, :]

    return label(output)


def ras2vec(classified_image, transform):
    classified_image = classified_image[0]
    props = regionprops(classified_image)

    shps = features.shapes(classified_image.astype(np.int32), connectivity=4,
                           transform=transform)
    shapes = list(shps)
    records = []

    count = 1
    for shp in shapes:
        if shp[1] != 0:
            id = int(shp[1]-1)
            count = np.int(props[id].area)
            perimeter = np.float(props[id].perimeter)
            eccentricity = np.float(props[id].eccentricity)
            equal_diam = np.float(props[id].equivalent_diameter)
            major_axis = np.float(props[id].major_axis_length)
            minor_axis = np.float(props[id].minor_axis_length)
            orientation = np.float(props[id].orientation)

            item = {'geometry': shp[0],
                    'id': count,
                    'properties': OrderedDict([('DN', np.int(shp[1])),
                                               ('count', count),
                                               ('perimeter', perimeter),
                                               ('eccentrici', eccentricity),
                                               ('equal_diam', equal_diam),
                                               ('major_axis', major_axis),
                                               ('minor_axis', minor_axis),
                                               ('orientatio', orientation)]),
                    'type': 'Feature'}
            records.append(item)
        count += 1

    return records


def add_zonal_fields(vector, raster, affine, prefix, band=0, stats=['mean',
                                                                    'min',
                                                                    'max',
                                                                    'std']):
    """
    Adds zonal statistics to an existing vector file.

    Add more documentation.

    """
    raster_stats = zonal_stats(vector, raster[band], stats=stats,
                               affine=affine)
    for item in raster_stats:
        items = tuple(item.items())
        for k, v in items:
            item[prefix + "_" + k] = v
            item.pop(k)

    for v, rs in zip(vector, raster_stats):
        v['properties'] = OrderedDict(v['properties'], **rs)

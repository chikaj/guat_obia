#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the summary line.

This script copies all the training data to a new folder.
Then it adjusts the x- and y-coordinates
so that they are located in the middle of
the study area and can receive training polygons

@author: Nate Currit
"""

import time
import rasterio
from rasterio import features
from affine import Affine
import numpy as np
from glob import glob
from pprint import pprint
import fiona
#from shapely.geometry import asShape, mapping
from collections import OrderedDict, Iterable
from rasterstats import zonal_stats
# import matplotlib
# import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.measure import regionprops, label
from skimage.filters import sobel


training_data = {
    'farming': [
        'new_IMG_3451',
        'new_IMG_3609',
        'new_IMG_3611',
        'new_IMG_3616',
        'new_IMG_3620',
        'new_IMG_3636',
        'new_IMG_3639',
        'new_IMG_3646',
        'new_IMG_3674',
        'new_IMG_3687',
        'new_IMG_3689',
        'new_IMG_3754',
        'new_IMG_4099',
        'new_IMG_4169',
        'new_IMG_4220',
        'new_IMG_4567',
        'new_IMG_4568',
        'new_IMG_4577',
        'new_IMG_4934',
        'new_IMG_5048',
        'new_IMG_5227'
    ],
    'ranching': [
        'new_IMG_3691',
        'new_IMG_3661',
        'new_IMG_3731',
        'new_IMG_3738',
        'new_IMG_4196',
        'new_IMG_4267',
        'new_IMG_4279',
        'new_IMG_4330',
        'new_IMG_4456',
        'new_IMG_4487',
        'new_IMG_4497',
        'new_IMG_4502',
        'new_IMG_4519',
        'new_IMG_4538',
        'new_IMG_4565',
        'new_IMG_4720',
        'new_IMG_4760',
        'new_IMG_4778',
        'new_IMG_4810',
        'new_IMG_4822',
        'new_IMG_4829',
        'new_IMG_4839',
        'new_IMG_5003',
        'new_IMG_5004',
        'new_IMG_5033',
        'new_IMG_5046',
        'new_IMG_5105',
        'new_IMG_5196',
        'new_IMG_5250',
        'new_IMG_5253',
        'new_IMG_5268',
        'new_IMG_5291',
        'new_IMG_5312',
        'new_IMG_5320'
    ],
    'guamil_alto': [
        'new_IMG_4331',
        'new_IMG_4524',
        'new_IMG_4563',
        'new_IMG_4668',
        'new_IMG_4763',
        'new_IMG_4816',
        'new_IMG_4825',
        'new_IMG_4993',
        'new_IMG_5052',
        'new_IMG_5058',
        'new_IMG_5175',
        'new_IMG_5286'
    ],
    'guamil_bajo': [
        'new_IMG_3746',
        'new_IMG_4460',
        'new_IMG_4525',
        'new_IMG_4562',
        'new_IMG_4574',
        'new_IMG_4781',
        'new_IMG_4784',
        'new_IMG_4811',
        'new_IMG_4824',
        'new_IMG_4931',
        'new_IMG_4935',
        'new_IMG_4989',
        'new_IMG_5050',
        'new_IMG_5065',
        'new_IMG_5210',
        'new_IMG_5244',
        'new_IMG_5287'
    ],
    'bosque_alto': [
        'new_IMG_3411',
        'new_IMG_3417',
        'new_IMG_3444',
        'new_IMG_3447',
        'new_IMG_3482',
        'new_IMG_3647',
        'new_IMG_3885',
        'new_IMG_3904',
        'new_IMG_4419',
        'new_IMG_4617',
        'new_IMG_4702',
        'new_IMG_4900',
        'new_IMG_5123'
    ],
    'bosque_bajo': [
        'new_IMG_3499',
        'new_IMG_3525',
        'new_IMG_3555',
        'new_IMG_3593',
        'new_IMG_3626',
        'new_IMG_3845',
        'new_IMG_4613',
        'new_IMG_4908',
        'new_IMG_4983'
    ],
    'wetlands': [
        'new_IMG_3456',
        'new_IMG_3569',
        'new_IMG_3581',
        'new_IMG_3700',
        'new_IMG_3743',
        'new_IMG_3813',
        'new_IMG_3833',
        'new_IMG_4043',
        'new_IMG_4246',
        'new_IMG_4533',
        'new_IMG_4551',
        'new_IMG_4636',
        'new_IMG_5028',
        'new_IMG_5194',
        'new_IMG_5205'
    ],
    'savanna': [
        'new_IMG_3460',
        'new_IMG_3595',
        'new_IMG_3596',
        'new_IMG_3794',
        'new_IMG_3812',
        'new_IMG_3862',
        'new_IMG_3990',
        'new_IMG_4002',
        'new_IMG_4102',
        'new_IMG_4232'
    ],
    'water': [
        'new_IMG_3414',
        'new_IMG_4087',
        'new_IMG_4240',
        'new_IMG_4486',
        'new_IMG_4535',
        'new_IMG_4594',
        'new_IMG_4602',
        'new_IMG_4870',
        'new_IMG_5041',
        'new_IMG_5170'
    ]
}

training_map = {'Farming': 1, 'Ranching': 2, 'Guamil Alto': 3,
                'Guamil Bajo': 4, 'Bosque Alto': 5, 'Bosque Bajo': 6,
                'Wetlands': 7, 'Sabana': 8, 'Brown water': 9,
                'Black water': 10, 'Yellow water': 11}


def get_training_data():
    """Return the training data."""
    return training_data


def time_elapsed(tm):
    hour, min, sec = 0, 0, 0
    diff = time.time() - tm
    if diff >= 3600:
        hour = diff // 3600
        diff = diff % 3600

    if diff >= 60:
        min = diff // 60
        diff = diff % 60

    sec = diff

    print("Time elapsed: " + str(int(hour)) + " hours, " +
          str(int(min)) + " minutes, " + str(sec) + " seconds.")


def flatten(ndim_list):
    """
    Flattens an n-dimensional list.

    Flattens, or converts, an n-dimensional list to a 1-dimensional list.

    Parameters
    ----------
    ndim_list: list
        N-dimensional array.

    Returns
    -------
    generator
        Returns a generator to the flattened list. Call list(generator) to get
        an actual list.

    """
    for parent in ndim_list:
        if isinstance(parent, Iterable) and not isinstance(parent, str):
            for child in flatten(parent):
                yield child
        else:
            yield parent


def shift_images():
    """
    Summary line. Shift the images' coordinates.

    Extended description of function. This function shifts the
    image coordinates and removes the 4th Alpha band from
    the photograph.

    Parameters
    ----------
    arg1: int
        Description of arg1
        ...except there are no arguments for this function!!

    Returns
    -------
    int
        Description of return value
        ...except this does not return anything.

    """
    x_coord = 476703
    y_coord = 1952787
    x_increment = 350
    y_increment = 350
    x_count = 0
    y_count = 0
    for key in training_data:
        y_shift = y_count * y_increment
        for f in training_data[key]:
            with rasterio.open(photo_path + f + ".tif") as src:
                x_shift = x_count * x_increment
                meta = src.meta
                meta['count'] = 3
                meta['transform'] = Affine.translation(x_coord + x_shift,
                                    y_coord - y_shift) * Affine.scale(af[0],
                                    af[4])
                with rasterio.open(photo_training_path + "training_" + f + ".tif", "w", **meta) as dst:
                    dst.write(src.read((1, 2, 3)))
            x_count += 1
        y_count += 1
        x_count = 0


def merge_training_data():
    """
    Merge all training photographs.

    Merge all training photographs into a single image.
    This is useful as a precursor to extracting raster
    statistics from a raster and then training a classifier.

    Parameters
    ----------
    none

    Returns
    -------
    success
        String

    """
    photos = glob.glob(photo_training_path + "*.tif")
    photo_readers = []
    for photo in photos:
        photo_readers.append(rasterio.open(photo))

    output = rasterio.merge.merge(photo_readers)
    meta = photo_readers[0].meta
    meta['width'] = output[0].shape[2]
    meta['height'] = output[0].shape[1]
    meta['transform'] = output[1]
    meta['nodata'] = 0
    try:
        with rasterio.open(photo_training_path + "big_training.tif", "w",
                           **meta) as dst:
            dst.write(output[0])

        return "Success!"
    except Exception:
        return "Failure"


def visualize_training_data():
    """
    Visualize training data.

    Visualize histograms and descriptive statistics of photographs.

    Parameters
    ----------
    none

    Returns
    -------
    none

    """
    col = []
    with rasterio.open(photo_training_path + "big_training.tif") as src:
        for i in range(1, src.count + 1):
            stats = rasterstats.zonal_stats(training_polygon_path +
                                            training_polygons, src.read(i),
                                            affine=src.transform)
            for s in range(0, len(stats)):
                col.append(stats[s]['mean'])

        return {"lines": np.array(col).reshape(3, 11), "histograms": ""}

    # next: visualize histograms with matplotlib...for each band and class.


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


def get_schema_definition(vector):
    geometry = vector[0]['geometry']['type']
    properties = vector[0]['properties']
    output = OrderedDict()
    for k, v in zip(properties.keys(), properties.values()):
        if isinstance(v, int):
            output[k] = 'int'
        elif isinstance(v, float):
            output[k] = 'float'
        else:
            output[k] = 'Datatype not read correctly'

    return {'geometry': geometry,
            'properties': output}


def main():
    in_path = "../training_data/"
    out_path = "./segs/"

    full_training_list = list(flatten(list(get_training_data().values())))
    training_list_firsts = ["training_new_IMG_3451",
                            "training_new_IMG_3691",
                            "training_new_IMG_4331",
                            "training_new_IMG_3746",
                            "training_new_IMG_3411",
                            "training_new_IMG_3499",
                            "training_new_IMG_3456",
                            "training_new_IMG_3460",
                            "training_new_IMG_3414"]
    single_image = ["training_new_IMG_3411"]
    lst = full_training_list

    start_time = time.time()
    counter = 1
    for l in lst:
        with rasterio.open(in_path + "training_" + l + ".tif") as src:
#        with rasterio.open(in_path + l + ".tif") as src:
            image = src.read(masked=True)
            mask = src.read_masks(1)
            # Segment the image
            r_output = slic_segmentation(image.transpose(1, 2, 0), mask)

            # Peform raster-to-vector conversion (in memory), including
            # appending shape metrics for each polygon
            v_output = ras2vec(r_output, transform=src.transform)

            # Add zonal statistics for the spectral bands of each segment
            for b, p in zip(range(0, src.count), ['red', 'green', 'blue']):
                add_zonal_fields(vector=v_output, raster=image, band=b,
                                 affine=src.transform, prefix=p,
                                 stats=['mean'])

            # Perform edge detection
            edges = sobel(image[0])
            edges = edges[np.newaxis, :]
            # Add zonal statistics for texture (i.e., mean, std edges)
            # of each segment
            add_zonal_fields(vector=v_output, raster=edges, band=0,
                             affine=src.transform, prefix='sobel',
                             stats=['mean', 'std', 'sum'])

#            Comment/Uncomment the following lines to (de)activate code
#            to write the segmented (1) raster and (2) vector to disk.
#            Be sure to properly set the output names.
#            ---(1)-Raster-----------------------------------------------------
#            r_meta = src.meta
#            r_meta['count'] = 1
#            r_meta['dtype'] = 'int32'
#            with rasterio.open("raster_output.tif", "w", **r_meta) as dst:
#                dst.write(r_output.astype('int32'))
#            ---(2)-Vector-----------------------------------------------------
            v_meta = {'driver': 'ESRI Shapefile', 'crs': src.crs.to_dict(),
                      'schema': get_schema_definition(v_output)}

            with fiona.open(out_path + l + "_slico3.shp", "w", **v_meta) as dst:
                dst.writerecords(v_output)

            pprint("Completed " + str(counter) + " out of " + str(len(lst)) +
                   " iterations.")
            time_elapsed(start_time)
            counter += 1

    time_elapsed(start_time)


if __name__ == "__main__":
    main()

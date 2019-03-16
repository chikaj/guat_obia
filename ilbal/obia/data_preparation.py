#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:49:31 2018

@author: nate
"""

from . import utility
import rasterio
from rasterio import features
from geopandas import GeoDataFrame
from affine import Affine
from glob import glob
import numpy as np
from collections import OrderedDict
from rasterstats import zonal_stats
#from skimage.segmentation import slic, felzenszwalb, quickshift
from skimage.measure import regionprops, label
from skimage.filters import sobel
from skimage.morphology import disk
from skimage.filters.rank import modal


def is_working():
    print("data_preparation.py is working!")


def shift_images(input_directory_path, output_directory_path):
    """
    Shift the images' coordinates.

    This function shifts the image coordinates and removes the 4th Alpha band
    from the photograph. It is used to center photographs from distant
    locations to a central location for manual digitizing for purposes of
    training an image classifier.

    Parameters
    ----------
    input_directory_path: string
        string representing filesystem directory where the photographs are
        located.
    
    output_directory_path: string
        string representing the filesystem directory where the output
        photographs will be written to disk.

    Returns
    -------
    NADA

    """
    x_coord = 476703
    y_coord = 1952787
    x_increment = 350
    y_increment = 350
    x_count = 0
    y_count = 0
    training_data = utility.get_training_data()
    for key in training_data:
        y_shift = y_count * y_increment
        for f in training_data[key]:
            with rasterio.open(input_directory_path + f + ".tif") as src:
                x_shift = x_count * x_increment
                af = src.transform
                meta = src.meta
                meta['count'] = 3
                meta['transform'] = Affine.translation(x_coord + x_shift,
                                    y_coord - y_shift) * Affine.scale(af[0],
                                    af[4])
                with rasterio.open(output_directory_path + "training_" + f + ".tif", "w", **meta) as dst:
                    dst.write(src.read((1, 2, 3)))
            x_count += 1
        y_count += 1
        x_count = 0


def merge_training_data(directory_path):
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
    photos = glob.glob(directory_path + "*.tif")
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
        with rasterio.open(directory_path + "big_training.tif", "w",
                           **meta) as dst:
            dst.write(output[0])

        return "Success!"
    except Exception:
        return "Failure"


# Expects a rasterio.io.DataSource
def rc_to_wh(image):
    return image.transpose(0, 2, 1)


def wh_to_rc(image):
    return image.transpose(0, 2, 1)


def bsq_to_bip(image):
    # no error checking yet...
    return  image.transpose(1, 2, 0)


def bip_to_bsq(image):
    # no error checking yet...
    return  image.transpose(2, 0, 1)


def bsq_to_bil(image):
    # no error checking yet...
    return  image.transpose(1, 0, 2)


def bil_to_bsq(image):
    # no error checking yet...
    return  image.transpose(1, 0, 2)


def bip_to_bil(image):
    # no error checking yet...
    return  image.transpose(0, 2, 1)


def bil_to_bip(image):
    # no error checking yet...
    return  image.transpose(0, 2, 1)


#def slic_segmentation(image, mask):
#    """
#    Segment the image.
#
#    Segment the image using the slic algorithm (from sklearn.segmentation).
#
#    Parameters
#    ----------
#    image: numpy.array
#        A rasterio-style image. Obtained and transformed by:
#        src.read(masked=True).transpose(1, 2, 0)
#
#    mask: numpy.array
#        A rasterio-style mask. Obtained by src.read_masks(1)
#        This function doesn't do anything with mask at the moment.
#        This function assumes image has, and is read with, a mask.
#
#    Returns
#    -------
#    numpy.array
#        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
#        so it's ready to be written by rasterio
#
#    """
#    img = image[0:3].transpose(1, 2, 0)
#    segs = slic(img, n_segments=5000, compactness=1, slic_zero=False)
#    mask[mask == 255] = 1
#    output = segs * mask[np.newaxis, :]
#
#    return label(output)
#
#
#def slic_zero_segmentation(image, mask):
#    """
#    Segment the image.
#
#    Segment the image using the slic-0 algorithm (from sklearn.segmentation).
#
#    Parameters
#    ----------
#    image: numpy.array
#        A rasterio-style image. Obtained and transformed by:
#        src.read(masked=True).transpose(1, 2, 0)
#
#    mask: numpy.array
#        A rasterio-style mask. Obtained by src.read_masks(1)
#        This function doesn't do anything with mask at the moment.
#        This function assumes image has, and is read with, a mask.
#
#    Returns
#    -------
#    numpy.array
#        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
#        so it's ready to be written by rasterio
#
#    """
#    img = image[0:3].transpose(1, 2, 0)
#    segs = slic(img, n_segments=2500, compactness=0.1, sigma=5, slic_zero=True)
#    mask[mask == 255] = 1
#    output = segs * mask[np.newaxis, :]
#
#    return label(output)


def segmentation(model=None, params=None, src=None, bands=[1,2,3], image=None,
                 mask=None, modal_radius=None, sieve_size=250):
    """
    Segment the image.

    Segment the image using the felzenszwalb algorithm
    (from sklearn.segmentation).

    Parameters
    ----------
    src: Rasterio datasource
        A rasterio-style datasource, created using:
        with rasterio.open('path') as src.
        There must be at least 3 bands of image data. If there are more than
        3 bands, the first three will be used. This parameter is optional.
        If it is not provided, then image and transform must be supplied.
    
    bands: array of integers
        The array of 3 bands to read from src as the RGB image for segmentation.
        
    image: numpy.array
        A 3-band (RGB) image used for segmentation. The shape of the image
        must be ordered as follows: (bands, rows, columns).
        This parameter is optional.
    
    mask: numpy.array
        A 1-band image mask. The shape of the mask must be ordered as follows:
        (rows, columns). This parameter is optional.
    
    transform: rasterio.transform
        A raster transform used to convert row/column values to geographic
        coordinates. This parameter is optional.
    
    modal_radius: integer
        Integer representing the radius of a raster disk (i.e., circular
        roving window). Optional. If not set, no modal filter will be applied.
    
    sieve_size: integer
        An integer representing the smallest number of pixels that will be
        included as a unique segment. Segments this size or smaller will be
        merged with the neighboring segment with the most pixels. 

    Returns
    -------
    numpy.array
        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
        so it's ready to be written by rasterio

    """
    if src is not None:
        img = bsq_to_bip(src.read(bands, masked=True))
        mask = src.read_masks(1)
        mask[mask > 0] = 1
    else:
        img = bsq_to_bip(image)
        mask[mask > 255] = 1
    
    output = model(img, **params).astype('int32')

    while np.ndarray.min(output) < 1:
        output += 1

    if modal_radius != None:
        output = modal(output.astype('int16'), selem=disk(modal_radius),
                       mask=mask)

    output = features.sieve(output, sieve_size, mask=mask,
                            connectivity=8) * mask
    output = label(output, connectivity=2)
    
    output = bip_to_bsq(output[:, :, np.newaxis]) * mask

    return output


#def felz_segmentation(src=None, bands=[1,2,3], image=None, mask=None,
#                      modal_radius=None, sieve_size=250):
#    """
#    Segment the image.
#
#    Segment the image using the felzenszwalb algorithm
#    (from sklearn.segmentation).
#
#    Parameters
#    ----------
#    src: Rasterio datasource
#        A rasterio-style datasource, created using:
#        with rasterio.open('path') as src.
#        There must be at least 3 bands of image data. If there are more than
#        3 bands, the first three will be used. This parameter is optional.
#        If it is not provided, then image and transform must be supplied.
#    
#    bands: array of integers
#        The array of 3 bands to read from src as the RGB image for segmentation.
#        
#    image: numpy.array
#        A 3-band (RGB) image used for segmentation. The shape of the image
#        must be ordered as follows: (bands, rows, columns).
#        This parameter is optional.
#    
#    mask: numpy.array
#        A 1-band image mask. The shape of the mask must be ordered as follows:
#        (rows, columns). This parameter is optional.
#    
#    transform: rasterio.transform
#        A raster transform used to convert row/column values to geographic
#        coordinates. This parameter is optional.
#    
#    modal_radius: integer
#        Integer representing the radius of a raster disk (i.e., circular
#        roving window). Optional. If not set, no modal filter will be applied.
#    
#    sieve_size: integer
#        An integer representing the smallest number of pixels that will be
#        included as a unique segment. Segments this size or smaller will be
#        merged with the neighboring segment with the most pixels. 
#
#    Returns
#    -------
#    numpy.array
#        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
#        so it's ready to be written by rasterio
#
#    """
#    if src is not None:
#        img = bsq_to_bip(src.read(bands, masked=True))
#        mask = src.read_masks(1)
#        mask[mask > 0] = 1
#    else:
#        img = bsq_to_bip(image)
#        mask[mask > 255] = 1
#    
#    output = felzenszwalb(img, scale=10.0, sigma=2,
#                          min_size=5000).astype('int32')
#
#    while np.ndarray.min(output) < 1:
#        output += 1
#
#    if modal_radius != None:
#        output = modal(output.astype('int16'), selem=disk(modal_radius),
#                       mask=mask)
#
#    output = features.sieve(output, sieve_size, mask=mask,
#                            connectivity=8) * mask
#    output = label(output, connectivity=2)
#    
#    output = bip_to_bsq(output[:, :, np.newaxis]) * mask
#
#    return output
#
#
#def quickshift_segmentation(image, mask):
#    """
#    Segment the image.
#
#    Segment the image using the quickshift algorithm
#    (from sklearn.segmentation).
#
#    Parameters
#    ----------
#    image: numpy.array
#        A rasterio-style image. Obtained and transformed by:
#        src.read(masked=True).transpose(1, 2, 0)
#
#    mask: numpy.array
#        A rasterio-style mask. Obtained by src.read_masks(1)
#        This function doesn't do anything with mask at the moment.
#        This function assumes image has, and is read with, a mask.
#
#    Returns
#    -------
#    numpy.array
#        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
#        so it's ready to be written by rasterio
#
#    """
#    img = image[0:3].transpose(1, 2, 0)
#    segs = quickshift(img, ratio=0.3, kernel_size= 2, max_dist=50, sigma=5)
#    mask[mask == 255] = 1
#    output = segs * mask[np.newaxis, :]
#
#    return label(output, connectivity=1)


def get_prop(props, label):
    for p in props:
        if p.label == label:
            return p


def vectorize(src=None, image=None, transform=None):
    """
    Raster-to-Vector conversion.
    
    Performs a raster-to-vector conversion of a classified image. 
    
    Parameters
    ----------
    src: Rasterio datasource
        A rasterio-style datasource created using: 
            with rasterio.open('path') as src.
        The datasource referred to must be a classified image.
        This parameter is optional. If it is not provided then the 
        image and the transform must be provided. 

    image: numpy.array
        A signle band of (classified, ideally) image data where the pixel
        values are integers. Shape is (1, rows, columns). This parameter is
        optional.
    
    transform: rasterio.transform
        A raster transform used to convert row/column values to geographic
        coordinates. This parameter is optional.
    
    Returns
    -------
    GeoDataFrame
        A vector version of the classified raster.
    """
    if src is not None:
        img = src.read(1, masked=True)
        transform = src.transform
    else:
        img = image[0].astype(np.int32)
        
    shps = features.shapes(img, connectivity=8, transform=transform)
    records = []

    for id, shp in enumerate(shps):
        if shp[1] != 0:
            item = {'geometry': shp[0], 'id': id+1, 'properties': 
                    OrderedDict([('dn', np.int(shp[1]))]),
                    'type': 'Feature'}
            records.append(item)

    return GeoDataFrame.from_features(records)


def _geometry_value_pairs(gdf, value_column):
    """
    Converts GeoDataFrame to (geometry, value) pairs
    
    Parameters
    ----------
    gdf: GeoDataFrame
        A GeoDataFrame from which is extracted the geometry and specified
        value for each geographic object (i.e., point, line, polygon). 
    
    value_column: string
        A string identifying the column to use as values.
    
    Returns
    -------
    pairs: A list of geometry, value pairs. 
    """
    pairs = []
    geo = gdf.geometry
    val = gdf[value_column]
    for g, v in (zip(geo, val)): 
        pairs.append((g, v))
    
    return pairs


def rasterize(gdf, value_column, shape, transform):
    """
    Rasterizes a GeoDataFrame
    
    Rasterizes a GeoDataFrame where the value_column becomes the pixel id.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        A GeoDataFrame (i.e., vector) to rasterize.
        
    value_column: string
        A string representing the field (column) name used to create the
        raster values.
    
    shape: tuple of (rows, columns)
        The number of rows and columns of the output raster.
    
    transform: Affine transform
        An Affine Transform used to relate pixel locations to ground positions.
    
    Returns
    -------
    image: A rasterized version of the GeodataFrame in Rasterio order:
        [bands][rows][columns]
    """
    p = _geometry_value_pairs(gdf, value_column)
    image = features.rasterize(p, out_shape=shape, transform=transform)
    
    return image


def add_shape_properties(classified_image, gdf, attributes=['area', 'perimeter']):
    """
    Add raster properties as vector fields.
    
    POSSIBLE IMPROVEMENT!! REMOVE PARAMETER classified_image AND INSTEAD USE 
    rasterize TO RASTERIZE THE gdf. 
        
    Parameters
    ----------
    classified_image: numpy.array
        A 2D image with integer, class-based, values.
    
    gdf: GeoDataFrame
        A GeoDataFrame (vector) with object boundaries corresponding to image
        regions. Image attributes will be assigned to each vector object.
    
    shapes: list of strings
        Shapes is a list of strings where each string is a type of shape to
        calculate for each polygon. Possible shapes include: area, bbox_area,
        centroid, convex_area, eccentricity, equivalent_diamter, euler_number,
        extent, filled_area, label, maxor_axis_length, max_intensity,
        mean_intensity, min_intensity, minor_axis_length, orientation,
        perimeter, or solidity.
    
    Returns
    -------
    Nothing
        Instead modifies GeoDataFrame in place.
    """    
    props = regionprops(classified_image)
    
    attributes = {s: [] for s in attributes}

    for row in gdf.itertuples():
        r = row[1]
        p = get_prop(props, r)
        if p is not None:
            for a in attributes:
                attributes[a].append(getattr(p, a))
                
    for a in attributes:
        if (a == 'area'):
            gdf.insert(len(gdf.columns)-1, a, gdf.geometry.area)
        elif (a == 'perimeter'):
            gdf.insert(len(gdf.columns)-1, a, gdf.geometry.length)
        else:
            gdf.insert(len(gdf.columns)-1, a, attributes[a])
    
    return gdf



def add_zonal_properties(src=None, bands=[1,2,3], image=None, transform=None,
                            band_names=['red','green','blue'], stats=['mean'],
                            gdf=None):
    """
    Adds zonal properties to a GeoDataFrame.
    
    Adds zonal properties to a GeoDataFrame, where the statistics 'stats' are
    calculated for all pixels within the geographic objects boundaries.
    
    Parameters
    ----------
    src: Rasterio datasource
        A rasterio-style datasource created using: 
            with rasterio.open('path') as src.
        This parameter is optional. If it is not provided then the 
        image and the transform must be provided.
    
    bands: list of integers
        The list of bands to read from src. This parameter is optional if src
        is not provided.

    image: numpy.array
        A signle band of (classified, ideally) image data where the pixel
        values are integers. Shape is (1, rows, columns). This parameter is
        optional.
    
    transform: rasterio.transform
        A raster transform used to convert row/column values to geographic
        coordinates. This parameter is optional.
    
    band_names: list of strings
        The labels corresponding to each band of the src or image. 
    
    stats: list of strings
        The list of zonal statistics to calculate for each geographic object.
    
    gdf: GeoDataFrame
        The GeoDataFrame to be updated with zonal statistics. The number of
        columns is equal to len(bands) * len(stats). 
    
    Returns
    -------
    GeoDataFrame
        A GeoDataFrame with the zonal statistics added as new columns. 
    """
    if src is not None:
        image = src.read(bands, masked=True)
        transform = src.transform

    if len(image) != len(band_names): 
        print("The number of image bands must equal the number of bands passed.")
        return None

    for b, p in enumerate(band_names):
        raster_stats = zonal_stats(gdf, image[b], stats=stats, affine=transform)
        
        fields = [[] for i in range(len(stats))]
        labels = []
        
        for i, rs in enumerate(raster_stats):
            for j, r in enumerate(rs):
                if i == 0:
                    labels.append(r)
                fields[j].append(rs[r])
        
        for i, l in enumerate(labels):
            gdf.insert(len(gdf.columns)-1, p + "_" + l, fields[i])

    return gdf


def edge_detect(src=None, band=1, image=None, mask=None):
    """
    Performs a Sobel edge detection.

    Performs a Sobel edge detection on a 2D image.
    
    Parameters
    ----------
    src: Rasterio datasource
        A rasterio-style datasource created using: 
            with rasterio.open('path') as src.
        This parameter is optional. If it is not provided then image must be
        provided.
    
    band: integer
        The band to read from src. This band will be used for edge detection.

    image: numpy.array
        A rasterio-style image. The image is any single band obtained by: 
            image = src.read(band, masked=True), where band is an integer.
        This parameter is optional.
            
    mask: numpy.array
        A rasterio-style image. The image is any single band obtained by: 
            image = src.read_masks(1), where band is an integer. 
        This parameter is optional.
        
    Returns
    -------
    numpy.array
        A single band, rasterio-style image ([band][row][column]).
    """
    if src is not None:
        image = src.read(band, masked=True)
        mask = src.read_masks(1)
        mask[mask > 0] = 1
    else:
        image = image
        mask[mask > 255] = 1
        
    edges = sobel(image)
    return bip_to_bsq(edges[:, :, np.newaxis]) * mask


#def prep_for_slic(image, mask, transform, crs_dict):
#    rout = slic_segmentation(image, mask)
#    vout = ras2vec(rout, transform)
#    for b, p in zip(range(0, len(image[0:3])), ['red', 'green', 'blue']):
#        add_zonal_fields(vector=vout, raster=image, band=b, affine=transform,
#                         prefix=p, stats=['mean'])
#    edges = edge_detect(image[0])
#    add_zonal_fields(vector=vout, raster=edges, band=0,
#                     affine=transform, prefix='sobel',
#                     stats=['mean', 'std', 'sum'])
#    
#    return vout
#
#
#def prep_for_slic_zero(image, mask, transform, crs_dict):
#    rout = slic_zero_segmentation(image, mask)
#    vout = ras2vec(rout, transform)
#    for b, p in zip(range(0, len(image[0:3])), ['red', 'green', 'blue']):
#        add_zonal_fields(vector=vout, raster=image, band=b, affine=transform,
#                         prefix=p, stats=['mean'])
#    edges = edge_detect(image[0])
#    add_zonal_fields(vector=vout, raster=edges, band=0,
#                     affine=transform, prefix='sobel',
#                     stats=['mean', 'std', 'sum'])
#    
#    return vout
#
#
#def prep_for_felz(src):
#    rout = felz_segmentation(src, modal_radius=3, sieve_size=249)
#    vout = ras2vec(rout, src.transform)
#    vout = add_shape_properties(rout, vout)
#    vout = add_spectral_properties(src, ['red', 'green', 'blue'], vout)
#    
##    vout.to_file("sieved.shp")
#
#    image = src.read(masked=True)[0:3]
#    for b, p in zip(range(len(image)), ['red', 'green', 'blue']):
#        add_zonal_fields(vector=vout, raster=image, band=b, affine=src.transform,
#                         prefix=p, stats=['mean'])
#    edges = edge_detect(image[0])
#    add_zonal_fields(vector=vout, raster=edges, band=0,
#                     affine=src.transform, prefix='sobel',
#                     stats=['mean', 'std', 'sum'])
#    
#    return vout
#
#
#def prep_for_quickshift(image, mask, transform, crs_dict):
#    rout = quickshift_segmentation(image, mask)
#    vout = ras2vec(rout, transform)
#    for b, p in zip(range(0, len(image[0:3])), ['red', 'green', 'blue']):
#        add_zonal_fields(vector=vout, raster=image, band=b, affine=transform,
#                         prefix=p, stats=['mean'])
#    edges = edge_detect(image[0])
#    add_zonal_fields(vector=vout, raster=edges, band=0,
#                     affine=transform, prefix='sobel',
#                     stats=['mean', 'std', 'sum'])
#    
#    return vout

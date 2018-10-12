#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:47:34 2018

@author: nate
"""

import time
from collections import OrderedDict, Iterable
import fiona
import rasterio


def is_working():
    print("utility.py is working!")


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


def get_training_data():
    """Return the training data."""
    return training_data

def get_training_data_as_list():
    """Return the training data as a 1D list."""
    output = []
    for i in flatten(get_training_data()):
        output.append(get_training_data()[i])

    return flatten(output)


def time_elapsed(tm):
    """
    Prints the time elapsed.
    
    Prints the time elapsed since tm, in hours, minutes and seconds. 
    
    Parameters
    ----------
    tm: time
    
    Returns
    -------
    NADA, it prints the time elapsed. 
    
    """
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


def get_schema_definition(vector):
    """
    Returns the schema definition of an existing vector.

    Returns the schema definition of an existing vector. This is useful
    when creating new vectors with the same schema as an existing vector.

    Parameters
    ----------
    vector: list of dictionaries with items that internally (i.e., there is no
    Shapefile or other file type on disk) represent a vector file. 
        list of dictionary items used by Fiona: geometry, id, properties and
        type. These dictionary items mostly correspond to the GeoJSON format.

    Returns
    -------
    dictionary
        Returns a dictionary that defines the vector schema of the vector.

    """
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


def write_shapefile(vector, crs, filename):
    v_meta = {'driver': 'ESRI Shapefile', 'crs': crs,
              'schema': get_schema_definition(vector)}

    with fiona.open(filename, "w", **v_meta) as dst:
        dst.writerecords(vector)


def write_geotiff(raster, filename, matching_src=None, height=None, width=None,
                  count=None, dtypes=None, crs=None, transform=None):
    """
    Write a raster (formatted in rasterio (BSQ) style) as a GeoTIFF.
    
    Parameters
    ----------
    raster: numpy.ndarray
        A raster with in rasterio (BSQ) ordered format--[bands, rows, columns].
        
    filename: string
        The output filename, including extension.
    
    matching_src: Rasterio datasource
        A rasterio datasource obtained using: 
            with rasterio.open(filename) as src:
        This is an optional parameter. If it is not provided, then the remaining
        parameters must be set.
    
    height: integer
        Height of raster (i.e., the number of rows). Optional.

    width: integer
        Width of raster (i.e., the number of columns). Optional.

    count: integer
        The number of raster bands. Optional.
    
    dtype = tuple of strings
        A tuple of strings representing the raster's datatype (e.g., 'int32').
        Optional.
    
    crs: string
        A string representation of the coordinate reference system. Optional.
    
    transform: Affine transformation
        A rasterio style affine transform. Optional.
    """
    if height == None:
        height = matching_src.height
    if width == None:
        width = matching_src.width
    if count == None:
        count = matching_src.count
    if dtypes == None:
        dtypes = matching_src.dtypes
    if crs == None:
        crs = matching_src.crs.to_string()
    if transform == None:
        transform = matching_src.transform

    with rasterio.open(filename, "w", driver="GTiff", height=height, 
                       width=width, count=count, dtype=dtypes, crs=crs,
                       transform=transform) as dst:
        dst.write(raster)

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:34:29 2017

@author: nc17
"""

import rasterio
import fiona
from collections import OrderedDict

in_path = "../Users/nc17/Research/Guatemala/photos/2015/PNLT/output2/"
out_path = in_path + "boundaries/"

data = ['new_IMG_3451',
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
        'new_IMG_5227']

def get_data():
    return data


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


def outline(filenames):
    for filename in filenames:
        with rasterio.open(in_path + filename + ".tif") as src:
            image = src.read_masks(1)
            
            shps = rasterio.features.shapes(image, connectivity=8,
                                   transform=src.transform)
            records = []
            
            shapes = list(shps)
            for shape in shapes:
                if shape[1] != src.nodata:
                    item = {'geometry': shape[0],
                            'id': shape[0],
                            'properties': OrderedDict([('DN', shape[1])]),
                            'type': 'Feature'}
                    records.append(item)
                    
            v_meta = {'driver': 'ESRI Shapefile', 'crs': src.crs.to_dict(),
                      'schema': get_schema_definition(records)}
                    
            with fiona.open(out_path + filename + "_yipee2.shp", "w", **v_meta) as dst:
                dst.writerecords(records)


def raster_to_shapefile(raster, shapefile):
    with rasterio.open(raster) as src:
        v_meta = {'driver': 'ESRI Shapefile', 'crs': src.crs.to_dict(),
                  'schema': {'geometry': 'Polygon',
                             'properties': OrderedDict([('DN', 'int')])}}
        with fiona.open(shapefile, "w", **v_meta) as dst:
            image = src.read(1)
            shapes = rasterio.features.shapes(image, connectivity=8,
                                              transform=src.transform)
            for shape in shapes:
                if shape[1] != src.nodata:
                    item = {'geometry': shape[0],
                            'id': shape[0],
                            'properties': OrderedDict([('DN', shape[1])]),
                            'type': 'Feature'}
                    dst.write(item)
    
def main():
    outline(get_data())
    raster_to_shapefile(in_path + 'new_IMG_3451.tif', out_path + "1tester_3451.shp")

if __name__ == "__main__":
    main()
       
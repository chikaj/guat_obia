# -*- coding: utf-8 -*-
"""
Loop through the images to classify them.

@author: Nate Currit
"""
from ilbal.obia import data_preparation as dp, classify as cl
from ilbal.obia import verification as v
import rasterio
#import numpy as np
#import geopandas as gpd
#from geopandas import GeoDataFrame as gdf
#import pickle
from glob import glob
from skimage.segmentation import slic#, felzenszwalb
from skimage.future import graph
#import psycopg2
from sqlalchemy import create_engine
from geoalchemy2 import Geometry, WKTElement
import time
from os import environ


def _segment(filename):
    with rasterio.open(filename) as src:
        slic_params = {'compactness': 20,
                       'n_segments': 200,
                       'multichannel': True}

        # Segment the image.
        rout = dp.segmentation(model=slic, params=slic_params, src=src,
                               modal_radius=3)

        # Region Agency Graph to merge segments
#        orig = dp.bsq_to_bip(src.read([1, 2, 3], masked=True))
#        labels = (dp.bsq_to_bip(rout))[:, :, 0]
#
#        rag = graph.rag_mean_color(orig, labels, mode='similarity')
#        rout = graph.cut_normalized(labels, rag)
#
#        # Vectorize the RAG segments
#        rout = dp.bip_to_bsq(rout[:, :, np.newaxis])
        vout = dp.vectorize(image=rout, transform=src.transform,
                            crs=src.crs.to_proj4())

        # Add spectral properties.
        vout = dp.add_zonal_properties(src=src, bands=[1, 2, 3],
                                       band_names=['red', 'green', 'blue'],
                                       stats=['mean','min','max','std'],
                                       gdf=vout)

#        t = vout.drop(['dn', 'geometry'], axis=1) # may be used later...

        # Add shape properties.
        vout = dp.add_shape_properties(rout, vout, ['area', 'perimeter',
                                                    'eccentricity', 
                                                    'equivalent_diameter',
                                                    'major_axis_length',
                                                    'minor_axis_length',
                                                    'orientation'])

        # Add texture properties.
        edges = dp.sobel_edge_detect(src, band=1)
        vout = dp.add_zonal_properties(image=edges, band_names=['sobel'],
                                       stats=['mean','min','max','std'],
                                       transform=src.transform, gdf=vout)

        ###################
        # Temporary code to write to vector and raster formats...
        # for checking output. Akin to using print statements to debug.
        #vout.to_file("output/working.shp")
        #out_raster = dp.rasterize(vout, 'dn', src.shape, src.transform)
        #utility.write_geotiff(out_raster[np.newaxis, :], 'rasterized.tif',
        #                       src, count=1, dtypes=('uint8'))
        ###################

        return vout


def segment(image_list):
    connection_string = environ.get('guat_obia_connection_string',
                                    'postgresql://nate:nate@localhost:5432/guat_obia')
    engine = create_engine(connection_string)

    for i, image in enumerate(image_list):
        vout = _segment(image)

        vout['geom'] = vout['geometry'].apply(lambda x: WKTElement(x.wkt, srid=9001))
        vout.drop('geometry', 1, inplace=True)
        vout.to_sql('training', engine, 'nate', if_exists='append', index=False,
                    dtype={'geom': Geometry('POLYGON', srid=9001)})


if __name__ == "__main__":
    import os
    if not os.path.exists("output"):
        os.makedirs("output")

    ##### Set the location #####
    location = "local" # "local" or "txgisci"
    if location == "local":
        training_path = "/home/nate/Documents/Research/Guatemala/training/"
    else:
        training_path = "/data1/guatemala/training/"
    ############################

    image_list = glob(training_path + "training_new_IMG_*.tif")

    startTime = time.time()
    segment(image_list)
    endTime = time.time()
    print("The segmentation took " + str(endTime - startTime) + " seconds to complete.")

#    startTime = time.time()
#    train()
#    endTime = time.time()
#    print("The training took " + str(endTime - startTime) + " seconds to complete.")

#    test()
#    verify()

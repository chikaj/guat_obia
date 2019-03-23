# -*- coding: utf-8 -*-
"""
Loop through the images to classify them.

@author: Nate Currit
"""
from ilbal.obia import data_preparation as dp, classify as cl
from ilbal.obia import verification as v
#import numpy as np
import rasterio
import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame as gdf
#import pandas as pd
#from skimage.filters import sobel
import pickle
from glob import glob

from skimage.segmentation import felzenszwalb, slic
from skimage.future import graph


def segment(filename):
    with rasterio.open(filename) as src:
#        felz_params = {'scale': 100.0,
#                       'sigma': 2,
#                       'min_size': 5000}

        slic_params = {'slic_zero': True}

        # Segment the image.
#        rout = dp.segmentation(model=felzenszwalb, params=felz_params, src=src,
#                               modal_radius=3)
        rout = dp.segmentation(model=slic, params=slic_params, src=src,
                               modal_radius=3)
        ##### temporary
        vout = dp.vectorize(image=rout, transform=src.transform)
        vout.to_file("output/original_segs.shp")
        #####

        # Region Agency Graph to merge segments
        orig = dp.bsq_to_bip(src.read([1, 2, 3], masked=True))
        rag_segs = (dp.bsq_to_bip(rout))[:, :, 0]

        g = graph.rag_mean_color(orig, rag_segs)
        rout = graph.cut_normalized(rag_segs, g)

        # Vectorize the RAG segments
        rout = dp.bip_to_bsq(rout[:, :, np.newaxis])
        vout = dp.vectorize(image=rout, transform=src.transform)

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
        vout.to_file("output/working.shp")
#        out_raster = dp.rasterize(vout, 'dn', src.shape, src.transform)
#        utility.write_geotiff(out_raster[np.newaxis, :], 'rasterized.tif',
#                              src, count=1, dtypes=('uint8'))
        ###################

        return vout


def train():    
    location = "local" # "local" or "txgisci"
    if location == "local":
        training_path = "/home/nate/Documents/Research/Guatemala/training_data/"
    else:
        training_path = "/data1/Guatemala/PNLT2015/need/to/add/training/data"
        
    image_list = glob(training_path + "training_new_IMG_*.tif")
        
    # build training Geodataframe
    for i, image in enumerate(image_list):
        if i == 0:
            vout = segment(image)
        else:
            vout = vout.append(segment(image))

    # select objects intersecting training polygons from...
    training_polys = "/home/nate/Documents/Research/Guatemala/geobia_training/training_samples2.shp"
    vpoly = gpd.read_file(training_polys)
    
    # old stuff...I'm working on it...
    for i in image_list:
        vout = felz(i)

        gdf.to_file(vout, i.replace(".tif", "_segs_f.shp"))
    
        fields = ['count', 'perimeter', 'eccentrici', 'equal_diam',
                  'major_axis', 'minor_axis', 'orientatio', 'red_mean',
                  'green_mean', 'blue_mean', 'sobel_mean', 'sobel_std']
#        tmp = vec.loc[:, fields]
        tmp = vout[fields]
        to_predict = tmp.values
        predictions = cl.predict(model, to_predict)
        
        vout['pred'] = predictions
        vout.crs = src.crs.to_dict()

        verification = gpd.read_file(verif)
        full = gpd.sjoin(vout, verification, how="inner", op="within")
        
        cls = full['class_id'].values
        prd = full['pred'].values
        o, p, u, k = v.accuracy(v.cross_tabulation(cls, prd, 11))
        
        print("--------------------------------------------------")
        print()
        print("For image " + i + ", the accuracy metrics are:")
        print("\tOverall Accuracy: " + str(o) + "%")
        print("\tProducers Accuracy: " + str(p) + "%")
        print("\tUsers Accuracy: " + str(u) + "%")
        print("\tKappa Coefficient: " + str(k) + "%")
        print()
        
        gdf.to_file(full, i.replace(".tif", "_verify_segs.shp"))


def original_main():
#    model_path = "/home/nate/Documents/Research/Guatemala/guat_obia/felz_model"
#    model = pickle.load(open(model_path, "rb"))
#    print(model)
    
#    if argv[1] not in ('train', 'test', 'verify'):
#        sys.exit("Requires an argument from the following list: train, test, verify.")
    
    location = "local" # "local" or "txgisci"
    if location == "local":
        training_path = "/home/nate/Documents/Research/Guatemala/training_data/"
#        image_path = "/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/"
    else:
        training_path = "/data1/Guatemala/PNLT2015/"
        
#    image_list = glob(image_path + "*.tif")
    
    image_list = [training_path + 'new_IMG_3434.tif',
                  training_path + 'new_IMG_3435.tif',
                  training_path + 'new_IMG_3436.tif']
    
#    /home/nate/Documents/Research/Guatemala/geobia_training/training_samples2.shp
        
#    verif = image_path + 'verification.shp'

    # build training dataset
    for i, image in enumerate(image_list):
        if i == 0:
            vout = felz(image)
        else:
            vout = vout.append(felz(image))        
    
    # old stuff...I'm working on it...
    for i in image_list:
        vout = felz(i)

        gdf.to_file(vout, i.replace(".tif", "_segs_f.shp"))
    
        fields = ['count', 'perimeter', 'eccentrici', 'equal_diam',
                  'major_axis', 'minor_axis', 'orientatio', 'red_mean',
                  'green_mean', 'blue_mean', 'sobel_mean', 'sobel_std']
#        tmp = vec.loc[:, fields]
        tmp = vout[fields]
        to_predict = tmp.values
        predictions = cl.predict(model, to_predict)
        
        vout['pred'] = predictions
        vout.crs = src.crs.to_dict()

        verification = gpd.read_file(verif)
        full = gpd.sjoin(vout, verification, how="inner", op="within")
        
        cls = full['class_id'].values
        prd = full['pred'].values
        o, p, u, k = v.accuracy(v.cross_tabulation(cls, prd, 11))
        
        print("--------------------------------------------------")
        print()
        print("For image " + i + ", the accuracy metrics are:")
        print("\tOverall Accuracy: " + str(o) + "%")
        print("\tProducers Accuracy: " + str(p) + "%")
        print("\tUsers Accuracy: " + str(u) + "%")
        print("\tKappa Coefficient: " + str(k) + "%")
        print()
        
        gdf.to_file(full, i.replace(".tif", "_verify_segs.shp"))


if __name__ == "__main__":
    import os
    if not os.path.exists("output"):
        os.makedirs("output")
    train()
#    test()
#    verify()

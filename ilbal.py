# -*- coding: utf-8 -*-
"""
Loop through the images to classify them.

@author: Nate Currit
"""
from ilbal.obia import data_preparation as dp, classify as cl
from ilbal.obia import verification as v, utility
#import numpy as np
import rasterio
import geopandas as gpd
from geopandas import GeoDataFrame as gdf
#import pandas as pd
#from skimage.filters import sobel
import pickle
from glob import glob

from skimage.segmentation import slic, felzenszwalb
from sklearn.cluster import DBSCAN


def felz(filename):
    with rasterio.open(filename) as src:
        felz_params = {'scale': 100.0,
                       'sigma': 2,
                       'min_size': 5000}
        
        slic_params = {'n_segments': 150,
                        'compactness': 0.1,
                        'sigma': 5,
                        'slic_zero': True}

        rout = dp.segmentation(model=felzenszwalb, params=felz_params, src=src,
                               modal_radius=3)
        vout = dp.vectorize(image=rout, transform=src.transform)
        vout = dp.add_zonal_properties(src=src, bands=[1, 2, 3],
                                       band_names=['red', 'green', 'blue'],
                                       stats=['mean','min','max','std'],
                                       gdf=vout)
        # perform clustering (k-means?) on SLIC0 segments
        t = vout.drop(['dn', 'geometry'], axis=1)
        clusters = DBSCAN().fit_predict(t)
        # re-do zonal properties for spectral bands, then continue
        vout = dp.add_shape_properties(rout, vout, ['area', 'perimeter',
                                                    'eccentricity', 
                                                    'equivalent_diameter',
                                                    'major_axis_length',
                                                    'minor_axis_length',
                                                    'orientation'])
        edges = dp.edge_detect(src, band=1)
        vout = dp.add_zonal_properties(image=edges, band_names=['sobel'],
                                       stats=['mean','min','max','std'],
                                       transform=src.transform, gdf=vout)
        vout.to_file("output/working.shp")
#        out_raster = dp.rasterize(vout, 'dn', src.shape, src.transform)
#        utility.write_geotiff(out_raster[np.newaxis, :], 'rasterized.tif',
#                              src, count=1, dtypes=('uint8'))
        return vout


def main():
#    model_path = "/home/nate/Documents/Research/Guatemala/guat_obia/felz_model"
#    model = pickle.load(open(model_path, "rb"))
#    print(model)
    
    image_path = "/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/"
    
    image_list = glob(image_path + "*.tif")
    
    image_list = ['/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3434.tif',
                  '/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3435.tif',
                  '/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3436.tif']
    
#    image_list = ['/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/new_IMG_3435.tif']
    
    verif = '/home/nate/Documents/Research/Guatemala/photos/2015/PNLT/output2/verification.shp'

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
    main()

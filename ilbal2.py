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
import pickle
from glob import glob
from skimage.segmentation import slic#, felzenszwalb
from skimage.future import graph
from multiprocessing import Process, Queue
import time
from math import ceil


def segment(filename):
    with rasterio.open(filename) as src:
#        felz_params = {'scale': 100.0,
#                       'sigma': 2,
#                       'min_size': 5000}

        slic_params = {'compactness': 20,
                       'n_segments': 200,
                       'multichannel': True}

        # Segment the image.
#        rout = dp.segmentation(model=felzenszwalb, params=felz_params, src=src,
#                               modal_radius=3)
        rout = dp.segmentation(model=slic, params=slic_params, src=src,
                               modal_radius=3)

        vout = dp.vectorize(image=rout, transform=src.transform,
                            crs=src.crs.to_proj4())
#        ##### temporary
#        vout.to_file("output/original_segs.shp")
#        #####

        # Region Agency Graph to merge segments
        orig = dp.bsq_to_bip(src.read([1, 2, 3], masked=True))
        labels = (dp.bsq_to_bip(rout))[:, :, 0]

        rag = graph.rag_mean_color(orig, labels, mode='similarity')
        rout = graph.cut_normalized(labels, rag)

        # Vectorize the RAG segments
        rout = dp.bip_to_bsq(rout[:, :, np.newaxis])
        vout = dp.vectorize(image=rout, transform=src.transform, crs=src.crs)

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
#        vout.to_file("output/working.shp")
        #out_raster = dp.rasterize(vout, 'dn', src.shape, src.transform)
        #utility.write_geotiff(out_raster[np.newaxis, :], 'rasterized.tif',
        #                       src, count=1, dtypes=('uint8'))
        ###################

        return vout


def train(image_list):
    # build training Geodataframe
    vout = gpd.GeoDataFrame()
    for i, image in enumerate(image_list):
        vout = vout.append(segment(image))

    return vout


def doWork(list, q):
    output = train(list)

    q.put(output)


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

    image_list = glob(training_path + "training_new_IMG_*.tif")
    ##### Set the number of cores #####
    cores = 3
    partition = []
    image_list = image_list[:3]
    size = ceil(len(image_list) / cores)
    for id, core in enumerate(range(cores)):
        partition.append(image_list[id * size:(id + 1) * size])

    startTime = time.time()

    vout = gpd.GeoDataFrame()
    for part in partition:
        vout = vout.append(train(part), ignore_index=True)

    vout.to_file("output/full_training.shp")

    endTime = time.time()
    #calculate the total time it took to complete the work
    workTime =  endTime - startTime

    #print results
    print("The job took " + str(workTime) + " seconds to complete")

#    #mark the start time
#    startTime = time.time()
#    #create a Queue to share results
#    q = Queue()
#    #create sub-processes to do the work
#    for part in partition:
#        p = Process(target=doWork, args=(part, q))
#        p.start()
#     
#    results = []
#    #grab values from the queue, one for each process
#    for c in range(cores):
#        #set block=True to block until we get a result
#        results.append(q.get(True))
#     
#    #append all results together
#    vout = results[0]
#    for c in range(cores) - 2:
#        c += 1
#        vout.append(results[c])
#     
##    p1.join()
##    p2.join()
##    p3.join()
##    p4.join()
##    p5.join()
##    p6.join()
##    p7.join()
##    p8.join()
#             
#    #mark the end time
#    endTime = time.time()
#    #calculate the total time it took to complete the work
#    workTime =  endTime - startTime
#     
#    #print results
#    print("The job took " + str(workTime) + " seconds to complete")
    
#    train()
#    test()
#    verify()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:32:15 2017

@author: Nate Currit
"""

import geopandas as gpd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import datetime
import pickle
import matplotlib.pyplot as plt
from pprint import pprint


def assign_actual(segments_path, training_path):
    """
    Assign land cover classes to training data.
    
    From a geospatial (polygon) vector file of image segments and their
    attributes, do 3 things:
        (1) Add a field named 'random' and assign a random value between
            0 and 1 to each record (segment).
        (2) Add a field named 'actual' that will store the integer
            representation of the actual land cover class for each segment.
        (3) Select segments that are completely within manually digitized
            land cover polygons used to define training data, and... 
        (4) Assign the appropriate land cover (integer) code for each segment
            in the 'actual' field. 

    Parameters
    ----------
    segments_path: string
        The string is the path to an ESRI Shapefile, read with GeoPandas
        into a GeoDataFrame. This Shapefile contains the image segments. 

    training_path: string
        The string is the path to an ESRI Shapefile, read with GeoPandas
        into a GeoDataFrame. This Shapefile contains the manually digitized
        training data.

    Returns
    -------
    gdf: GeoPandas GeoDataFrame
        A GeoDataFrame with the attributes for classification and the 'random'
        and 'actual' fields added.

    """
    pass


def train_classifier(segments, output_filename, fields=['count', 'orientation',
                    'red_mean', 'green_mean', 'blue_mean'], 
                    actual='class_id'):
    """
    Train classification algorithm.
    
    Train the Support Vector Machine classification algorithm using the
    specified fields. 

    Parameters
    ----------
    segments: GeoPandas GeoDataFrame
        A GeoDataFrame with the attributes for classification and the 'random'
        and 'actual' fields added.

    output_filename: string
        Output filename of the pickled trained SVM model.

    fields: list of strings
        A list of the fields used to perform the classification.

    actual: string
        A string representing the field representing the actual class to which
        the segment belongs. 

    Returns
    -------
    model: svm.SVC
        Returns a trained SVM model that can be used to classify other data.

    """
    random_pct = 0.7
    training = segments.loc[(segments.class_id != 0) &
                            (segments.random > random_pct), fields]

    training_class = segments.loc[(segments.class_id != 0) &
                                  (segments.random > random_pct), [actual]]
    
    plt.figure()
    training.plot.hist(bins=25)

    X = training.values
    Y = training_class.values.reshape(-1)

#    clf = svm.SVC()
#    clf.fit(X, Y)
#    pprint(vars(clf))
#    pickle.dump(clf, open(output_filename, "wb"))
#    svm_pred = clf.predict(X)
    
    gnb = GaussianNB()
    gnb.fit(X, Y)
    print(vars(gnb))
    gnb_pred = gnb.predict(X)
    
    return clf


def predict(model, segments, fields=['count', 'orientation', 'red_mean',
                                     'green_mean', 'blue_mean']):
    """
    Classify segments using SVM model

    Classify image segments using the trained Support Vector Machine model. 

    Parameters
    ----------
     model: svm.SVC
        A trained SVM model that can be used to classify other data.

    segments: GeoDataFrame
        Unclassified vector (polygon) segments.

    output_filename: string
        Output filename of the classified segments (as an ESRI Shapefile).

    """
    segs = segments[fields]

    predictions = model.predict(segs)

    return predictions


def assess_accuracy(actual, classified):
    """
    Accuracy Assessment
    
    Assess the accuracy of the classified image.

    Parameters
    ----------
    actual: numpy.array
        A single dimension array (i.e., vector) containing integers indicative
        of the "actual" (ground reference or "ground truth") land-cover
        classes.

    classified: numpy.array
        A single dimension array containing integers indicative of the 
        classified land-cover classes.

    Returns
    -------
    numpy.array
        A numpy array arranged as rasterio would read it (bands=1, rows, cols)
        so it's ready to be written by rasterio

    """
    pass


def main():
    src = gpd.read_file('./segs/training_segments.shp')
    
    fields = ['count', 'perimeter', 'eccentrici', 'equal_diam', 'major_axis',
              'minor_axis', 'orientatio', 'red_mean', 'green_mean',
              'blue_mean', 'sobel_mean', 'sobel_sum', 'sobel_std']
    
    svm_t = train_classifier(src, 'trained_svm_' +
                             str(datetime.datetime.now()).replace(" ", "_"),
                             fields, 'class_id')

    pprint(vars(svm_t))
    
#    new_model = pickle.load(open("trained_svm_2018-03-02_14:33:47.039663", "rb"))

    predictions = predict(svm_t, src, fields)
    src['best_guess'] = predictions
    src.to_file('whatever.shp')
    
#tr = src.loc[src.class_id != 0, src.columns.difference(['DN', 'class_id', 'training', 'testing', 'class', 'photo', 'best_guess', 'random', 'geometry'])]
#tr = src.loc[(src.class_id != 0) & (src.random > 0.85), src.columns.difference(['DN', 'class_id', 'training', 'testing', 'class', 'photo', 'best_guess', 'random', 'geometry'])]
#tr_id = src.loc[src.class_id != 0, ['class_id']]

if __name__ == "__main__":
    main()

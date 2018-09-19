#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:32:15 2017

@author: Nate Currit
"""

import numpy as np
import geopandas as gpd
from scipy.stats import expon
from sklearn import svm
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.naive_bayes import GaussianNB
import datetime
import pickle
#import matplotlib.pyplot as plt
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


def train_classifier_old(segments, output_filename, fields=['count', 'orientation',
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

    X = training.values
    Y = training_class.values.reshape(-1)

#    clf = svm.SVC()
#    clf.fit(X, Y)
#    pprint(vars(clf))
#    pickle.dump(clf, open(output_filename, "wb"))
#    svm_pred = clf.predict(X)
    
#    scores = cross_val_score(clf, X, Y, cv=5)
    
    # specify parameters and distributions to sample from
#    param_dist = {'C': expon(scale=100),
#                  'gamma': expon(scale=.1),
#                  'kernel': ['rbf'],
#                  'class_weight':['balanced', None]}

    # run randomized search
#    n_iter_search = 20
#    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                   n_iter=n_iter_search)
#
#    random_search.fit(X, Y)
#    pprint(vars(random_search))
#    pickle.dump(random_search, open(output_filename, "wb"))
#    svm_pred = random_search.predict(X)
    
    # run optimized classifier
    best_clf = svm.SVC(C=14.344592902738631, cache_size=200, class_weight=None,
                   coef0=0.0, decision_function_shape='ovr', degree=3,
                   gamma=7.694015754766104e-05, kernel='rbf', max_iter=-1,
                   probability=False, random_state=None, shrinking=True,
                   tol=0.001, verbose=False)
    best_clf.fit(X, Y)
    pprint(vars(best_clf))
    pickle.dump(best_clf, open(output_filename, "wb"))
    svm_pred = best_clf.predict(X)
    
    crosstab = cross_tabulation(Y, svm_pred)
    print(crosstab)
    
    return best_clf


def train_classifier(X, Y, output_filename):
    """
    Train classification algorithm.
    
    Train the Support Vector Machine classification algorithm using the
    specified fields. 

    Parameters
    ----------
    X: Pandas DataFrame
        A DataFrame with the objects used for training the classifier.
        Each object has the same number of fields used for training. 
    
    Y: Pandas DataFrame
        A DataFrame with a single field representing the class id. 

    output_filename: string
        Output filename of the pickled trained SVM model. 

    Returns
    -------
    clf: svm.SVC
        Returns a trained SVM model that can be used to classify other data.

    """

    clf = svm.SVC(C=14.344592902738631, cache_size=200, class_weight=None,
                   coef0=0.0, decision_function_shape='ovr', degree=3,
                   gamma=7.694015754766104e-05, kernel='rbf', max_iter=-1,
                   probability=False, random_state=None, shrinking=True,
                   tol=0.001, verbose=False)
    clf.fit(X, Y)
    pprint(vars(clf))
    pickle.dump(clf, open(output_filename, "wb"))
    
    return clf


def predict(model, X):
    """
    Classify segments using SVM model

    Classify image segments using the trained Support Vector Machine model. 

    Parameters
    ----------
     model: svm.SVC
        A trained SVM model that used to classify objects.

    X: Pandas DataFrame
        A DataFrame with the objects used for predicting Y values.
        Each object has the same number of fields used for training.

    Returns
    -------
    Y: Pandas DataFrame
        A DataFrame with the predicted class id for each object.

    """

    predictions = model.predict(X)

    return predictions


def cross_tabulation(actual, classified):
    """
    Accuracy Assessment
    
    Assess the accuracy of the classified image.

    Parameters
    ----------
    actual: numpy.array
        A single dimension array (i.e., vector) containing integers indicative
        of the "actual" (ground reference or "ground truth") land-cover
        classes. The values are expected to start with 1 and be sequential
        (i.e., 1, 2, 3, 4...not 1, 3, 4, 5...and not 0, 1, 2, 3).

    classified: numpy.array
        A single dimension array containing integers indicative of the 
        classified land-cover classes. The values are expected to be in the 
        same range as for the the variable 'actual'.

    Returns
    -------
    numpy.array
        A 2D numpy array where the rows represent the classified segments
        and the columns represent the actual segment classes

    """
    crosstab = np.zeros((np.unique(actual).size, np.unique(actual).size))
    for i in range(0, actual.size):
        crosstab[classified[i]-1][actual[i]-1] += 1
    
    total = 0
    diagonal = 0
    for r in range(0, crosstab.shape[0]):
        for c in range(0, crosstab.shape[1]):
            total += crosstab[r][c]
            if (r == c):
                diagonal += crosstab[r][c]
    print("The overall accuracy is: " + str(diagonal / total * 100) + "%")
    
    return crosstab


def main():
    base_path = '../../Documents/Research/Guatemala/guat_obia/'
    training_data = base_path + 'segs/training_segments.shp'
    src = gpd.read_file(training_data)
    
    classification_fields = ['count', 'perimeter', 'eccentrici', 'equal_diam', 'major_axis',
              'minor_axis', 'orientatio', 'red_mean', 'green_mean',
              'blue_mean', 'sobel_mean', 'sobel_sum', 'sobel_std']
    class_id_field = ['class_id']
    
    training_fraction = 0.50 # Percentage of objects in the training set
    s = src.query(class_id_field[0] + ' != 0') # Select objects with a class id
    t = s.sample(frac=training_fraction).sort_index() # Training objects
    v = s[~s.isin(t)].dropna(subset=[classification_fields[0]]) # Verification objects
    training_X = t[classification_fields] # Training objects with correct fields
    training_Y = t[class_id_field].values.reshape(-1).astype(int) # Training objects with correct class id field
    verification_X = v[classification_fields] # Verification objects with correct fields
    verification_Y = v[class_id_field].values.reshape(-1).astype(int) # Verification objects with correct class id field
    
#    svm_t = train_classifier_old(src, 'trained_svm_' +
#                             str(datetime.datetime.now()).replace(" ", "_"),
#                             fields, 'class_id')
#    pprint(vars(svm_t))
    
    svm_t = train_classifier(training_X, training_Y, 'trained_svm_' + 
                             str(datetime.datetime.now()).replace(" ", "_"))
    
#    pickled_model = base_path + 'trained_svm_2018-03-19_21:51:24.707615'
#    new_model = pickle.load(open(pickled_model, "rb"))
#    pprint(vars(new_model))
#
#    random_pct = 0.7
#    verification = src.loc[(src.class_id != 0) &
#                            (src.random < random_pct), fields]
#    verification_class = src.loc[(src.class_id != 0) & (src.random < random_pct),
#                             ['class_id']]
#    
#    X = verification.values
#    Y = verification_class.values.reshape(-1)
#
#    predictions = predict(new_model, X)
    
    predictions = predict(svm_t, verification_X)

    crosstab = cross_tabulation(verification_Y, predictions)
    print(crosstab)

    src['best_guess'] = predictions
    src.to_file('whatever.shp')
    

if __name__ == "__main__":
    main()

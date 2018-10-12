#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:50:26 2018

@author: nate
"""

def is_working():
    print("classify.py is working!")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:32:15 2017

@author: Nate Currit
"""

from scipy.stats import expon
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
import pickle


def train(segments, actual, output_filename):
    """
    Train classification algorithm.
    
    Train the Support Vector Machine classification algorithm using the
    specified fields. 

    Parameters
    ----------
    segments: numpy 2D array
        A 2D numpy array where there is one row for each segment and each
        column represents an attribute of the segments. 

    actual: numpy 1D array
        A 1D numpy array equal in length to the number of records in segments.
        The single column contains actual class values for each of the
        segments.

    output_filename: string
        Output filename of the pickled trained SVM model.

    Returns
    -------
    model: svm.SVC
        Returns a trained SVM model that can be used to classify other data.

    """
    clf = svm.SVC()
        
    # specify parameters and distributions to sample from
    param_dist = {'C': expon(scale=100),
                  'gamma': expon(scale=.1),
                  'kernel': ['rbf'],
                  'class_weight':['balanced', None]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

    random_search.fit(segments, actual) # this may take time...
    pickle.dump(random_search, open(output_filename, "wb"))
    
    return clf


def predict(model, segments):
    """
    Classify segments using a trained SVM model

    Classify image segments using the trained Support Vector Machine model. 

    Parameters
    ----------
     model: svm.SVC
        A trained SVM model that can be used to classify other data.

    segments: numpy 2D array
        A 2D numpy array where there is one row for each segment and each
        column represents an attribute of the segments. Identical to segments
        from the train_classifier function.
    """
    predictions = model.predict(segments)

    return predictions

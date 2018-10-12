# -*- coding: utf-8 -*-
"""
Verification Tools

These tools are used to perform verification analysis of remotely sensed
image classifications. 

@author: Nate Currit
"""
import numpy as np


def cross_tabulation(actual, predicted, num_classes):
    """
    Cross-tabulation
    
    Creates a cross-tabulation table from an image with the actual classes
    identified and an image with the predicted classes identified.

    Parameters
    ----------
    actual: numpy.array
        A single dimension array (i.e., vector) containing integers indicative
        of the "actual" (ground reference or "ground truth") land-cover
        classes. The values are expected to start with 1 and be sequential
        (i.e., 1, 2, 3, 4...not 1, 3, 4, 5...and not 0, 1, 2, 3).

    predicted: numpy.array
        A single dimension array containing integers indicative of the 
        predicted land-cover classes. The values are expected to be in the 
        same range as for the the variable 'actual'.

    Returns
    -------
    numpy.array
        A 2D numpy array where the rows represent the predicted segments
        and the columns represent the actual segment classes.

    """
#    crosstab = np.zeros((np.unique(predicted).size, np.unique(predicted).size))
#    for i in range(0, predicted.size):
#        crosstab[predicted[i]-1][actual[i]-1] += 1
    
    crosstab = np.zeros((num_classes, num_classes))
    for i in range(0, actual.size):
        crosstab[predicted[i]-1][actual[i]-1] += 1
    
    return crosstab


def accuracy(crosstab):
    """
    Accuracy Assessment
    
    Assess the overall, producers, users and kappa accuracy measures.

    Parameters
    ----------
    crosstab: numpy.array
        A 2D numpy array where the rows represent the predicted segments
        and the columns represent the actual segment classes.

    Returns
    -------
    overall: double
    producers: double
    users: double
    kapps: double
        Each of the values returned is a measure of classification accuracy
        expressed as a percentage. 'overall' is the overall accuracy,
        'producer' is the producers accuracy, 'users' is the users accuracy,
        and 'kappa' is the kappa coefficient of agreement

    """
    total = 0
    diagonal = 0
    for r in range(0, crosstab.shape[0]):
        for c in range(0, crosstab.shape[1]):
            total += crosstab[r][c]
            if (r == c):
                diagonal += crosstab[r][c]

    overall = diagonal / total * 100
    
    row_total = np.sum(crosstab, 1)
    col_total = np.sum(crosstab, 0)
    
    producers = np.zeros(crosstab.shape[1])
    users = np.zeros(crosstab.shape[0])
    for i in range(row_total.shape[0]):
        if (col_total[i] == 0):
            producers[i] = 0
        else:
            producers[i] = crosstab[i][i] / col_total[i] * 100

        if (row_total[i] == 0):
            users[i] = 0
        else:
            users[i] = crosstab[i][i] / row_total[i] * 100

    kappa = 0
    sum_of_opposites = 0
    for i in range(crosstab.shape[0]):
        sum_of_opposites += row_total[i] * col_total[i]
    num = total * diagonal - sum_of_opposites
    den = total ** 2 - sum_of_opposites
    kappa = num / den

    return overall, producers, users, kappa

def main():
    ct = np.array([61, 9, 1, 4, 0, 11, 56, 3, 5, 0, 0, 3, 65, 7, 0, 0, 3, 7,
                   63, 2, 2, 0, 0, 0, 73]).reshape(5, 5)
    overall, producers, users, kappa = accuracy(ct)
    print("The overall accuracy is: " + str(overall))
    print("The producers accuracy is: " + str(producers))
    print("The users accuracy is: " + str(users))
    print("The kappa coefficient is: " + str(kappa))

if __name__ == "__main__":
    main()

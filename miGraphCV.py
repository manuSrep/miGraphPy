#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Functions to perform cross-validation with miGraph kernel method.

:author: Jan Lammel
:author: Manuel Tuschen
:date: 04.02.2016
:license: FreeBSD

Reference
---------
Zhi-Hua Zhou, Yu-Yin Sun, and Yu-Feng Li. 2009. Multi-instance learning by treating instances as non-I.I.D. samples. In Proceedings of the 26th Annual International Conference on Machine Learning (ICML '09). ACM, New York, NY, USA, 1249-1256. DOI=10.1145/1553374.1553534 http://doi.acm.org/10.1145/1553374.1553534

License
---------
Copyright (c) 2016, Jan Lammel, Manuel Tuschen
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""




from __future__ import division, absolute_import, print_function

import sys
import os
import glob
import random
from random import shuffle
import copy
import fnmatch
from random import shuffle

import numpy as np
import h5py as h5

from scipy import interp
from sklearn import svm, metrics, cross_validation
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, auc, precision_score, recall_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from miGraph import buildKernel, buildKernel_threading, normalize_data, estimate_gamma




def miGraphCV(bags, bagsLabels, gamma, delta=None, delta_method='global', dist_method='gaussian', C=10, k=10):
    '''
    Funktion to perfrom k-fold cross-validation.
    
    :param bags: [ndarray [n,d]]. List of all multiple instance bag.
    :param bagsLabels: [] List with labels for each bag.
    :param gamma: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :param delta: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :param delta_method: If delta is None determines the method how to estimate delta. Can be 'local' or 'global'.
                         'local' will determine a separate delta for each bag while 'global' uses the same delta for all.
    :param C: The SVM regularizer.
    :param k: The k-fold parameter.
    
    :return gamma: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :return delta: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :return delta_method: If delta is None determines the method how to estimate delta. Can be 'local' or 'global'.
                         'local' will determine a separate delta for each bag while 'global' uses the same delta for all.
    :return C: The SVM regularizer.
    :return accuracy: []. List of all accuracy values.
    :return precision: []. List of precision accuracy values.
    :return recall: []. List of all recall values.
    :return F1: []. List of all F1 scores.
    :return ROC: [[fpr, tpr, thr]]. List of all ROC values.
    :return precRec: [[pre, rec, thr]]. List of all precision-recall values.
    '''

    if delta is not None:
        delta_method = None

    # define our ouput parameters
    y_gt = []
    y_pred = []
    y_predProb = []
    
    accuracy = []
    tpr = []
    tnr = []
    precision = []
    npv = []
    recall = []
    F1 = []
    ROC = []
    precRec = []
    
    # shuffle bags and labels for arbitrary order of labels
    shuffIndices = range(len(bags))
    shuffle(shuffIndices)

    bags = [bags[i] for i in shuffIndices]
    bagsLabels = [bagsLabels[i] for i in shuffIndices]

    # get indices for training and test set for cross validation
    CVindex = cross_validation.KFold(len(bags), k)
    for trainIndices, testIndices in CVindex:

        # seperating data to test- and training-set
        X_train = [bags[i] for i in trainIndices]
        X_test  = [bags[i] for i in testIndices]
        y_train = [bagsLabels[i] for i in trainIndices]
        y_test  = [bagsLabels[i] for i in testIndices]

        # building up kernels
        miKernelTrain = buildKernel_threading(X_train, X_train, gamma=gamma, delta=delta, delta_method=delta_method, dist_method=dist_method)
        miKernelTest = buildKernel_threading(X_test, X_train, gamma=gamma, delta=delta, delta_method=delta_method, dist_method=dist_method)

        # train on SVM
        modelSvm = svm.SVC(kernel='precomputed', C=C, probability=True)
        modelSvm.fit(miKernelTrain, y_train)
        yPredTest = modelSvm.predict(miKernelTest)
        yProbTest = modelSvm.predict_proba(miKernelTest)
        
        # Ground Truth, predicted labels and associated probability
        y_gt.append(y_test)
        y_pred.append(yPredTest)
        y_predProb.append(yProbTest)
        
        # Accuracy
        accuracy.append(metrics.accuracy_score(y_test, yPredTest))
        
        yPredTest1 = np.array(yPredTest) == 1
        yPredTest0 = np.array(yPredTest) == 0
        yTest1 = np.array(y_test) == 1
        yTest0 = np.array(y_test) == 0
        # True positive count and True negative count
        tp = np.sum(np.logical_and(yPredTest1, yTest1)) # correct positives
        tn = np.sum(np.logical_and(yPredTest0, yTest0)) # correct negatives
        
        # False positive count and False negative count for calculation of npv
        fp = np.sum(np.logical_and(yPredTest1, yTest0)) # false positives
        fn = np.sum(np.logical_and(yPredTest0, yTest1)) # false negatives
        
        # True positive rate = Sensivity
        # True positive rate = 1 - false negative rate
        tpr.append(tp / np.sum(yTest1))
        
        # True negative rate = specificity
        # True negative rate = 1 - false positive rate
        tnr.append(tn / np.sum(yTest0))
        
        # Precision = positive predictive value
        precision.append(precision_score(y_test, yPredTest))
        
        # Negative predictive value (negative equivalent to precision)
        npv.append(tn / (tn + fn))
        
        # Recall
        recall.append(recall_score(y_test, yPredTest))

        #F1 Score
        F1.append(f1_score(y_test, yPredTest))

        # Precision, Recall Curve
        pre, rec, thr = precision_recall_curve(y_test, yProbTest[:,1], pos_label=1)
        precRec.append([pre, rec, thr])

        # ROC
        roc_fpr, roc_tpr, thr = roc_curve(y_test, yProbTest[:,1], pos_label=1)
        ROC.append([roc_fpr, roc_tpr, thr])

            
    return gamma, delta, delta_method, C, y_gt, y_pred, y_predProb, tpr, tnr, \
            accuracy, precision, npv, recall, F1, ROC, precRec 




def saveCV(filename, gamma, delta, delta_method, C, y_gt, y_pred, y_predProb, tpr, tnr, accuracy, precision, npv, recall, F1, ROC, precRec):
    '''
    Funktion to save data after k-fold cross-validation.
    
    :param filename: The filename to save to including the directory.
    :param gamma: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :param delta: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :param delta_method: If delta is None determines the method how to estimate delta. Can be 'local' or 'global'.
                         'local' will determine a separate delta for each bag while 'global' uses the same delta for all.
    :param C: The SVM regularizer.    
    :param accuracy: []. List of all accuracy values.
    :param precision: []. List of precision accuracy values.
    :param recall: []. List of all recall values.
    :param F1: []. List of all F1 scores.
    :param ROC: [[fpr, tpr, thr]]. List of all ROC values.
    :param precRec: [[pre, rec, thr]]. List of all precision-recall values.
    '''
    
    filename = os.path.expanduser(filename)
    
    if not(filename.split('.')[-1] == 'h5' or filename.split('.')[-1] == 'hdf5'):
        filename += '.hdf5'

    
    f = h5.File(filename, 'w')
        
    f.attrs.create('gamma', np.array([gamma]))
    if delta is None:
        f.attrs.create('delta', np.array([-1]))
    else:
        f.attrs.create('delta', np.array([delta]))
    if delta_method is None:
        f.attrs.create('delta_method', np.array([-1]))
    else:
        f.attrs.create('delta_method', np.array([delta_method]))
    f.attrs.create('C', np.array([C]))
    
    length = len(sorted(y_gt, key=len, reverse=True)[0])
    yGt = np.array([xi+[np.nan]*(length-len(xi)) for xi in y_gt])
    f.create_dataset('y_gt', data=yGt, dtype=np.float64)
        
    yPredList = [list(xi) for xi in y_pred]    
    length = len(sorted(yPredList, key=len, reverse=True)[0])
    yPred = np.array([xi+[np.nan]*(length-len(xi)) for xi in yPredList])
    f.create_dataset('y_pred', data=yPred, dtype=np.float64)
        
    yPredProbList = [list(xi) for xi in y_predProb]
    length = len(sorted(yPredProbList, key=len, reverse=True)[0])
    yPredProb = np.array([xi+[np.array([np.nan, np.nan])]*(length-len(xi)) for xi in yPredProbList])
    f.create_dataset('y_predProb', data=np.array(yPredProb), dtype=np.float64)
    
    f.create_dataset('tpr', data=np.array(tpr), dtype=np.float64)
    f.create_dataset('tnr', data=np.array(tnr), dtype=np.float64)
    f.create_dataset('accuracy', data=np.array(accuracy), dtype=np.float64)
    f.create_dataset('precision', data=np.array(precision), dtype=np.float64)
    f.create_dataset('npv', data=np.array(npv), dtype=np.float64)
    f.create_dataset('recall', data=np.array(recall), dtype=np.float64)
    f.create_dataset('F1', data=np.array(F1), dtype=np.float64)

    f_roc = f.create_group('ROC')
    f_prec = f.create_group('precRec')

    
    for k in range(len(accuracy)):
        k_group1 = f_roc.create_group(str(k)) 
        k_group1.create_dataset('fpr', data=ROC[k][0], dtype=np.float64)
        k_group1.create_dataset('tpr', data=ROC[k][1], dtype=np.float64)
        k_group1.create_dataset('thr', data=ROC[k][2], dtype=np.float64)
        
        k_group2 = f_prec.create_group(str(k)) 
        k_group2.create_dataset('pre', data=precRec[k][0], dtype=np.float64)
        k_group2.create_dataset('rec', data=precRec[k][1], dtype=np.float64)
        k_group2.create_dataset('thr', data=precRec[k][2], dtype=np.float64)
        
    f.close()




# loading data again
def loadCV(filename):
    '''
    Funktion to load data from k-fold cross-validation.
    
    :param filename: The filename the directory.
    
    :return gamma: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :return delta: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :return delta_method: If delta is None determines the method how to estimate delta. Can be 'local' or 'global'.
                         'local' will determine a separate delta for each bag while 'global' uses the same delta for all.
    :return C: The SVM regularizer.    
    :return accuracy: []. List of all accuracy values.
    :return precision: []. List of precision accuracy values.
    :return recall: []. List of all recall values.
    :return F1: []. List of all F1 scores.
    :return ROC: [[fpr, tpr, thr]]. List of all ROC values.
    :return precRec: [[prec, rec, thr]]. List of all precision-recall values.
    '''
    
    filename = os.path.expanduser(filename)


    f = h5.File(filename, 'r')
    gamma = f.attrs.get('gamma')[0]
    delta = f.attrs.get('delta')[0]
    if delta == -1:
        delta = None
    delta_method = f.attrs.get('delta_method')[0]
    if delta_method == -1:
        delta_method = None
    C = f.attrs.get('C')[0]
    
    y_gt = list(f['y_gt'][:])
    y_gt = [list(labels) for labels in y_gt]
    for labels in y_gt:
        for i, label in enumerate(labels):
            if (np.isnan(label)):
                del labels[i]
    
    y_pred = list(f['y_pred'][:])
    y_pred = [list(labels) for labels in y_pred]
    for labels in y_pred:
        for i, label in enumerate(labels):
            if (np.isnan(label)):
                del labels[i]
    
    y_predProb = list(f['y_predProb'][:])
    y_predProb = [x[~np.isnan(x)].reshape(int(x[~np.isnan(x)].shape[0] / 2), 2) for x in y_predProb]
    
    
    tpr = list(f['tpr'][:])
    tnr = list(f['tnr'][:])
    accuracy = list(f['accuracy'][:])
    precision = list(f['precision'][:])
    npv = list(f['npv'][:])
    recall = list(f['recall'][:])
    F1 = list(f['F1'][:])
    
    
    
    ROC = []
    precRec = []
    for k in range(len(accuracy)):
        ROC.append([f['ROC/{k}/fpr'.format(k=k)][:],f['ROC/{k}/tpr'.format(k=k)][:],f['ROC/{k}/thr'.format(k=k)][:]])
        precRec.append([f['precRec/{k}/pre'.format(k=k)][:],f['precRec/{k}/rec'.format(k=k)][:],f['precRec/{k}/thr'.format(k=k)][:]])

    return gamma, delta, delta_method, C, y_gt, y_pred, y_predProb, tpr, tnr, \
            accuracy, precision, npv, recall, F1, ROC, precRec




def plotPrecRec(precRec, result_dir=''):
    '''
    Function to plot precision recall curves

    :param precRec: [[pre, rec, thr]]. List of all precision-recall values.
    :param result_dir. The output directory.

    '''

    result_dir = os.path.expanduser(result_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Compute Precision-Recall and plot curve

    k = len(precRec)

    curves = np.zeros((k,1000))
    interpol = np.linspace(0, 1., 1000)
    aucs = np.zeros((k))

    for i,entry in enumerate(precRec):


        pre, rec, thr = entry

        curves[i] = interp(interpol, pre, rec)
        aucs[i] = auc(rec, pre)

    curves[:,0] = 1
    curves[:,-1] = 0
    pre_mean = np.mean(curves, axis=0)
    pre_std = np.std(curves, axis=0)
    rec_mean = interpol
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)


    # plot
    plt.figure()
    plt.plot(rec_mean, pre_mean, label = u'AUC = %0.3f ± %0.3f ' % (auc_mean, auc_std))
    plt.errorbar(rec_mean[::100], pre_mean[::100], yerr=pre_std[::100], fmt='o', color='black', ecolor='black')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(result_dir, 'PrecisionRecall_CV.png'))
    plt.close()





def plotROC(ROC, result_dir=''):
    '''
    Function to plot precision recall curves

    :param ROC: [[fpr, tpr, thr]]. List of all ROC values.
    :param result_dir: The output directory.

    '''

    result_dir = os.path.expanduser(result_dir)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Compute ROC and plot curve

    k = len(ROC)

    curves = np.zeros((k,1000))
    interpol = np.linspace(0, 1., 1000)
    aucs = np.zeros((k))

    for i,entry in enumerate(ROC):
        fpr, tpr, thr = entry
        curves[i] = interp(interpol, fpr, tpr)
        aucs[i] = auc(fpr, tpr)

    curves[:,0] = 0
    curves[:,-1] = 1
    tpr_mean = np.mean(curves, axis=0)
    tpr_std = np.std(curves, axis=0)
    fpr_mean = interpol
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)


    # plot
    plt.figure()
    plt.plot(fpr_mean, tpr_mean, label = u'AUC = %0.3f ± %0.3f ' % (auc_mean, auc_std))

    plt.errorbar(fpr_mean[::100], tpr_mean[::100], yerr=tpr_std[::100], fmt='o', color='black', ecolor='black')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(os.path.expanduser(os.path.join(result_dir, 'ROC_CV.png')))
    plt.close()




if __name__ == '__main__':
     print()
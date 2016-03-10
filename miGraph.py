#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Functions for building miGraph kernel.

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
from threading import Thread, Lock
from multiprocessing import cpu_count
import Queue
from math import ceil
from random import sample, seed


import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import pdist, cdist, squareform, sqeuclidean

lock = Lock()




def calcDistMatrix(bag, method='sqeuclidean', gamma=1.):
    '''
    This function calculates the inner bag distance matrix. This matrix represents a graph of connected instances
    Differently than stated in the publication, in their implementation Zhou et al. used not the gaussian distance
    but the squared euclidiean distance.

    :param bag: ndarray [n,D]. One multiple instance bag.
    :param method: Norm used for distance calculation.
    :param gamma: Parameter for RBF kernel for gaussian distance.

    :return distMat: ndarray [n,n].  Distance matrix
    '''

    n = bag.shape[0] # the number of instances

    if (method == 'sqeuclidean'):
        # squared euclidean norm
        bag_i_norm = np.sum(bag**2, axis=1)
        distMat = np.tile(bag_i_norm, [n,1]) + np.tile(bag_i_norm, [n,1]).transpose() -2*np.dot(bag, bag.transpose())
    elif (method == 'gaussian'):
        bag_i_norm = np.sum(bag**2, axis=1)
        distMat = np.tile(bag_i_norm, [n,1]) + np.tile(bag_i_norm, [n,1]).transpose() -2*np.dot(bag, bag.transpose())
        distMat = 1 - np.exp(-gamma * distMat)

    #distMat = squareform(pdist(bag, 'sqeuclidean'))

    return distMat




def calcRbfKernel(bag1, bag2, gamma):
    '''
    This function calculates an rbf kernel for instances between two bags.

    :param bag1: ndarray [n,d].  A multiple instance bag.
    :param bag2: ndarray [m,d].  A multiple instance bag.
    :param gamma: The normalizing parameter for the radial basis function.

    return: kMat: ndarray [n,m]. The between instances kernel function.
    '''

    n = bag1.shape[0] # the number of instances in bag 1
    m = bag2.shape[0] # the number of instances in bag 2

    # suared euclidean norm
    bag1_i_norm = np.sum(bag1**2, axis=1)
    bag2_i_norm = np.sum(bag2**2, axis=1)
    distMat = np.array(np.tile(bag1_i_norm, [m,1]).transpose() + np.tile(bag2_i_norm, [n,1]) - 2*np.dot(bag1, bag2.transpose()), dtype=np.float64)

    # radial basis function
    kMat = np.exp(-gamma * distMat)

    return kMat




def calcKernelEntry(bag1, bag2, weightMatrix1, weightMatrix2, gamma):
    '''
    This function calculates one kg kernel entry comparing two bags.
    Differently than stated in the publication, in their implementation Zhou et al. normalized by taking the squareroot
    of the sum over the edge coeficcients.

    :param bag1: ndarray [n,d].  A multiple instance bag.
    :param bag2: ndarray [m,d].  A multiple instance bag.
    :param gamma: The normalizing parameter for the radial basis function.

    return: kMat: ndarray [n,m]. The between instances kernel function.
    '''

    n = bag1.shape[0] # the number of instances in bag 1
    m = bag2.shape[0] # the number of instances in bag 2

    activeEdgesCount1 = np.sum(weightMatrix1, axis=1) # number of edges per instance
    activeEdgesCount2 = np.sum(weightMatrix2, axis=1) # number of edges per instance


    activeEdgesCoef1 = 1. / (activeEdgesCount1 + 1e-3)    # offset to avoid division by zero if e.g. just one instance in a bag
    activeEdgesCoef2 = 1. / (activeEdgesCount2 + 1e-3) 

    k = calcRbfKernel(bag1, bag2, gamma=gamma)

    k = np.tile(activeEdgesCoef1, [m,1]).transpose() * np.tile(activeEdgesCoef2, [n,1]) * k

    k = np.sum(k) / np.sqrt(np.sum(activeEdgesCoef1)) / np.sqrt(np.sum(activeEdgesCoef2))

    return k




def calcKernelEntry_threading(kernel, q):
    '''
    This function calculates one kg kernel entry comparing two bags using multiple threads.
    Differently than stated in the publication, in their implementation Zhou et al. normalized by taking the squareroot
    of the sum over the edge coeficcients.

    :param kernel: ndarray [N,M].  The (empty) kernel to calculate entry for.
    :param q: The multithreading queue with input parameters.

    return: None
    '''

    while True:
        i, j, bag1, bag2, weightMatrix1, weightMatrix2, gamma = q.get() # extract values from queue

        n = bag1.shape[0] # the number of instances in bag 1
        m = bag2.shape[0] # the number of instances in bag 2

        activeEdgesCount1 = np.sum(weightMatrix1, axis=1) # number of edges per instance
        activeEdgesCount2 = np.sum(weightMatrix2, axis=1) # number of edges per instance

        activeEdgesCoef1 = 1. / (activeEdgesCount1 + 1e-3)      # offset to avoid division by zero if e.g. just one instance in a bag
        activeEdgesCoef2 = 1. / (activeEdgesCount2 + 1e-3)

        k = calcRbfKernel(bag1, bag2, gamma=gamma)

        k = np.tile(activeEdgesCoef1, [m,1]).transpose() * np.tile(activeEdgesCoef2, [n,1]) * k

        k = np.sum(k) / np.sqrt(np.sum(activeEdgesCoef1)) / np.sqrt(np.sum(activeEdgesCoef2))
					
        with lock:
            kernel[i,j] = k

        q.task_done()




def buildKernel(bags1, bags2, gamma=None, delta=None, delta_method='global', dist_method='gaussian', progressBar=True):
    '''
    This function builds the final normalized miGraph kernel.

    :param bag1: array [N, n, d].  A set of multiple instance bag.
    :param bag2: array [M, m, d].  A set of multiple instance bag.
    :param gamma: The normalizing parameter for the radial basis function.
    :param delta: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :param delta_method: If delta is None determines the method how to estimate delta. Can be 'local' or 'global'.
                         'local' will determine a separate delta for each bag while 'global' uses the same delta for all.
    :param dist_method: Norm used for distance calculation.
    :param progressBar: Turn printing of progress bar on/off.

    return: kg: ndarray [N,M]. The between instances kernel function.
    '''

    N = len(bags1)
    M = len(bags2)

    kernel = np.zeros((N,M))

    distMatrices1 = []
    distMatrices2 = []
    dist_sum = 0
    dist_num = 0

    for i in range(N):
        distMatrices1.append(calcDistMatrix(bags1[i], method=dist_method, gamma=gamma))
        dist_sum += np.sum(distMatrices1[i])
        dist_num += distMatrices1[i].shape[0] * distMatrices1[i].shape[1]

    for i in range(M):
        distMatrices2.append(calcDistMatrix(bags2[i], method=dist_method, gamma=gamma))
        dist_sum += np.sum(distMatrices2[i])
        dist_num += distMatrices2[i].shape[0] * distMatrices2[i].shape[1]

    if delta is None and delta_method =='global':
        delta = dist_sum / dist_num


    diagNorm1 = np.zeros(N) # Contains normalization for each diagonal element
    weightMatrices1 = [] # List of weight matrices
    for i in range(N):
        if delta is None and delta_method =='local':
            delta = np.mean(distMatrices1[i])
            weightMatrices1.append((distMatrices1[i] < delta).astype(np.uint8))
            delta = None
        else:
            weightMatrices1.append((distMatrices1[i] < delta).astype(np.uint8))
        diagNorm1[i] = calcKernelEntry(bags1[i], bags1[i], weightMatrices1[i], weightMatrices1[i], gamma=gamma)


    diagNorm2 = np.zeros(M) # Contains normalization for each diagonal element
    weightMatrices2 = [] # List of weight matrices
    for i in range(M):
        if delta is None and delta_method =='local':
            delta = np.mean(distMatrices2[i])
            weightMatrices2.append((distMatrices2[i] < delta).astype(np.uint8))
            delta = None
        else:
            weightMatrices2.append((distMatrices2[i] < delta).astype(np.uint8))
        diagNorm2[i] = calcKernelEntry(bags2[i], bags2[i], weightMatrices2[i], weightMatrices2[i], gamma=gamma)

    T = N*M
    k = 0
    for i in range(N):
        for j in range(M):

            kernelEntry = calcKernelEntry(bags1[i], bags2[j], weightMatrices1[i], weightMatrices2[j], gamma=gamma)

            kernel[i,j] = kernelEntry / np.sqrt(diagNorm1[i] * diagNorm2[j])

            if ((k%(ceil(T/100)) == 0 or k == T-1) and progressBar == True):
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d%%" % ('='*int(float(k+1)/T*50), int(float(k+1)/T*100)))
                sys.stdout.flush()
            k += 1
    if (progressBar == True):
        sys.stdout.write('\n')

    return kernel



def buildKernel_threading(bags1, bags2, gamma=None, delta=None, delta_method='global', dist_method='gaussian', progressBar=True, n_threads=-1):
    '''
    This function builds the final normalized miGraph kernel.

    :param bag1: array [N, n, d].  A set of multiple instance bag.
    :param bag2: array [M, m, d].  A set of multiple instance bag.
    :param gamma: The normalizing parameter for the radial basis function.
    :param delta: The weight parameter to detemine when a given distance is regarded an edge. If None, the mean of
                  all distances is used.
    :param delta_method: If delta is None determines the method how to estimate delta. Can be 'local' or 'global'.
                         'local' will determine a separate delta for each bag while 'global' uses the same delta for all.
    :param dist_method: Norm used for distance calculation.
    :param progressBar: Turn printing of progress bar on/off.
    :param n_threads: The number of threads to use. If -1 the number of processors will be taken.

    return: kg: ndarray [N,M]. The between instances kernel function.
    '''

    N = len(bags1)
    M = len(bags2)

    kernel = np.zeros((N,M))

    distMatrices1 = []
    distMatrices2 = []
    dist_sum = 0
    dist_num = 0

    for i in range(N):
        distMatrices1.append(calcDistMatrix(bags1[i], method=dist_method, gamma=gamma))
        dist_sum += np.sum(distMatrices1[i])
        dist_num += distMatrices1[i].shape[0] * distMatrices1[i].shape[1]

    for i in range(M):
        distMatrices2.append(calcDistMatrix(bags2[i], method=dist_method, gamma=gamma))
        dist_sum += np.sum(distMatrices2[i])
        dist_num += distMatrices2[i].shape[0] * distMatrices2[i].shape[1]

    if delta is None and delta_method =='global':
        delta = dist_sum / dist_num


    diagNorm1 = np.zeros(N) # Contains normalization for each diagonal element
    weightMatrices1 = [] # List of weight matrices
    for i in range(N):
        if delta is None and delta_method =='local':
            delta = np.mean(distMatrices1[i])
            weightMatrices1.append((distMatrices1[i] < delta).astype(np.uint8))
            delta = None
        else:
            weightMatrices1.append((distMatrices1[i] < delta).astype(np.uint8))
        diagNorm1[i] = calcKernelEntry(bags1[i], bags1[i], weightMatrices1[i], weightMatrices1[i], gamma=gamma)


    diagNorm2 = np.zeros(M) # Contains normalization for each diagonal element
    weightMatrices2 = [] # List of weight matrices
    for i in range(M):
        if delta is None and delta_method =='local':
            delta = np.mean(distMatrices2[i])
            weightMatrices2.append((distMatrices2[i] < delta).astype(np.uint8))
            delta = None
        else:
            weightMatrices2.append((distMatrices2[i] < delta).astype(np.uint8))
        diagNorm2[i] = calcKernelEntry(bags2[i], bags2[i], weightMatrices2[i], weightMatrices2[i], gamma=gamma)


    if n_threads == -1:
        n_threads = cpu_count()

    threadQ = Queue.Queue()

    for i in range(n_threads):
        worker = Thread(target=calcKernelEntry_threading, args=(kernel, threadQ,))
        worker.setDaemon(True)
        worker.start()


    T = N*M
    k = 0
    for i in range(N):
        for j in range(M):

            threadQ.put((i, j, bags1[i], bags2[j], weightMatrices1[i], weightMatrices2[j], gamma))

            if ((k%(ceil(T/100)) == 0 or k == T-1) and progressBar):
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d%%" % ('='*int(float(k+1)/T*50), int(float(k+1)/T*100)))
                sys.stdout.flush()
            k += 1
    if progressBar:
        sys.stdout.write('\n')

    threadQ.join()


    diagNorm1 = np.tile(diagNorm1, [M, 1]).transpose()
    diagNorm2 = np.tile(diagNorm2, [N, 1])

    kernel /= np.sqrt(diagNorm1 * diagNorm2)


    return kernel




def normalize_data(bags, method='zeroMeanOneVar'):
    '''
    Normalize the data for the SVM

    :param bags: All the bags.
    :param method: Choose normalization method. Can be either 'zeroMeanOneVar' or 'zeroMinOneMax'.

    :return bags: All the normalized bags.
    '''

    instance_features = bags[0]

    for i in range(1,len(bags)):
        instance_features = np.concatenate((instance_features, bags[i]))

    if method == 'zeroMeanOneVar':
        instance_features = preprocessing.scale(instance_features, axis=0)
    elif method == 'zeroMinOneMax':
        min_max_scaler = preprocessing.MinMaxScaler()
        instance_features = min_max_scaler.fit_transform(instance_features)

        print(np.max(instance_features[:,0]), np.min(instance_features[:,0]))



    start = 0
    end = 0
    for i in range(len(bags)):
        end += bags[i].shape[0]
        bags[i] = instance_features[start:end]
        start = end

    return bags




def estimate_gamma(bags, method='med_dist'):
    '''
    Find a gamma estimate for the data

    :param bags: All the bags.
    :param method: The method to estimate sigma. Either 'med_dist', the median distance or 'num_feature' the number of features.

    :return gamma: The estimate for gamma.
    '''

    instance_features = bags[0]

    for i in range(1,len(bags)):
        instance_features = np.concatenate((instance_features, bags[i]))

    if method == 'num_feature':
        return 1/(instance_features.shape[1])

    if method == 'med_dist':
        n = instance_features.shape[0]
        if n > 10000: # If there are too many instances we can only take a subset
            seed(1)
            seq = sample(range(0,n), 10000)
            instance_features = instance_features[seq]

        return 1/(np.median(pdist(instance_features))**2)



if __name__ == '__main__':
     print()

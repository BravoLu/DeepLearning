from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from utils import to_numpy

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

def precision_at_k(relevance_score, k):
    # relevance_score: a list of relevance scores, e.g. [1, 0, 0, 1, 1, 1, 0]
    # return precision@k, 1 <= k <= len(relevance_score)
    # precision@4 = 2 / 4 = 0.5
    relevance_score = np.array(relevance_score, dtye = float)
    pak = relevance_score[k-1] * relevance_score[:k].sum() / k
    return pak

def average_precision(relevance_score, top_k = 1, epsilon = 0.00001):
    # ap([1, 0 ,0, 1, 1, 1, 0]) = 0.69
    relevance_score = relevance_score[:top_k]
    precision_list  = [precision_at_k(relevance_score, i) for i in range(1, top_k + 1)]
    ap = sum(precision_list) / (sum(relevance_score) + epsilon)
    return ap

def cmc(distmat, query_ids=None, gallery_ids=None,
         topk=100):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)

    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:,np.newaxis])
    # matches e.g
    # false false true true
    # true true false false
    ret = np.zeros(topk)
    
    for i in range(m):
        index = np.nonzero(matches[i])[0]
        delta = 1. / len(index)
        for j,k in enumerate(index):
            if k-j >= topk: break
            ret[k-j] += delta
        
    return ret.cumsum() / m
    

def mean_ap(distmat, query_ids=None, gallery_ids=None):
    distmat = to_numpy(distmat)      
    m, n = distmat.shape
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    aps = []

    for i in range(m):
        y_true = matches[i]
        y_score = -distmat[i][indices[i]]

        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)















  

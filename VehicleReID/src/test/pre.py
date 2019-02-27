# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-20 19:35:32
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-28 14:42:57
import numpy as np 

def precision_at_k(relevance_score , k):
	relevance_score = np.array(relevance_score, dtype = float)
	print(relevance_score[:5])
	pak = relevance_score[k-1] * relevance_score[:k].sum() / k
	return pak 


rel = [1, 1 , 0 , 0, 0, 0, 0]
print(precision_at_k(rel, 1))
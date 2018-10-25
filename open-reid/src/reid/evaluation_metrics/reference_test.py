# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-10-23 20:32:21
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-10-23 21:56:13

import numpy as np 
from collections import defaultdict 
from sklearn.metrics import average_precision_score

distmat = [[1,2,3,4],[3,2,4,1],[4,5,6,1],[2,8,4,1]]
query_ids = np.arange(4)
gallery_ids = np.arange(4)


indices = np.argsort(distmat,axis=1)
matches = (gallery_ids[indices] == query_ids[:,np.newaxis])

print('indices:\n{}\ngallery_ids[indices]:\n{}\nquery_ids[:,np.newaxis]:\n{}\nmatches:\n{}\n'
	.format(indices,gallery_ids[indices],query_ids[:,np.newaxis],matches))

for i in range(4):
	valid = gallery_ids[indices[i]] != query_ids[i]   #判断哪些是合理的index
	index = indices[i][valid]						  #取合理的index
	gids  = gallery_ids[index]						  #取合理index的id
	inds  = np.where(valid)[0] 						  #取valid为True的下标
	ids_dict = defaultdict(list)
	for j,x in zip(inds,gids):                        #x为gallery_id, j为index
		ids_dict[x].append(j)

	print('valid:\n{}\nindex:\n{}\ngids:\n{}\ninds:\n{}\n'
		.format(valid,index,gids,inds))
	print('ids_dict:{}'.
		format(ids_dict))
	#sampled = (valid & _unique_sample(ids_dict,len(valid)))
	#index2   = np.nonzero(matches[i,sampled])[0]

	# print('sampled:\n{}\nindex2:\n{}\n'.
		# format(sampled,index2))
	topk = 4
	repeat = 1 
	first_match_break = False
	ret = np.zeros(topk)
	for _ in range(repeat):	
		index2 = np.nonzero(matches[i,valid])[0]   
		print('matches[i,valid]:\n{}\nindex2:\n{}\n'.
			format(matches[i,valid],index2))
		# len(index2)取除去junk index，匹配的个数
		delta = 1. / (len(index2) * repeat)
		# index 第几个匹配
		# j为top(j+1)，k为匹配的index
		for j, k in enumerate(index2):
			if k - j >= topk: break
			if first_match_break:
				ret[k - j] += 1
				break
			ret[k - j] += delta

		print('ret:\n{}\n'
			.format(ret))
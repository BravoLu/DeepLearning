# Code Reference
## evaluation_metrics - 
### clasification.py
```
	accuracy(output,target,topk=(1,))
	#计算Top-k的准确率
```


### ranking.py
三个参数separate_camera_set, single_gallery_shot,first_match_break的作用分别是：
separate_camera_set:
single_gallery_shot:
first_match_break:
```
	cmc(distmat,
		query_ids=None,
		gallery_ids=None,
		query_cams=None,
		gallery_cams=None,
		topk=100,
		separate_camera_set=False,
		single_gallery_shot=False,
		first_match_break=False
		)
	'''
	m,n = distmat.shape - 

	distmat,(query,gallery)相似矩阵
	indices = np.argsort(distmat,axis=1) - 对横轴排序
	matches = (gallery_ids[indices] == query_ids[:,np.newaxis]) - 这里gallery_ids[indices]是一个矩阵维，所以query_ids加一个维度

	ret = np.zeros(topk) - 初始化
	num_valid_queries = 0 - 合理的queries数

	for i in range(m):  - 对每个query
	valid = ((gallery_ids[indices[i]]!=query_ids[i]) | (gallery_cams[indices[i]]!=query_cams[i])) - 过滤相同的id和相同的camera

	if separate_camera_set:
		valid &= (gallery_cams[indices[i]] != query_cams[i])

	if not np.any(matches[i,valid]): continue - 如果没有匹配

	if single_gallery_shot:
		repeat = 10
		gids = gallery_ids[indices[i][valid]] #取
		inds = np.where(valid)[0]
		ids_dict = defaultdict(list)
		for j,x in zip(inds, gids):
			ids_dict[x].append(j)
	else:
		repeat = 1
	
	for _ in range(repeat):
		if single_gallery_shot:
			~  
		else:
			index = np.nonzero(matches[i,valid])[0] - matches[i,valid]得到所有合理的index，nonzero取的是matches中True的值。
		delta = 1./ (len(index) * repeat)  - len(index)匹配的个数，index为匹配的下标，k-j为排名-k
		for j,k in enumerate(index):
			if k - j >= topk:break
			if first_match_break:
				ret[k-j] += 1
				break
			ret[k-j] += delta
	''' 
```


### mean_ap(distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None)
$AP = \sum_n(R_n-R_{n-1})P_n$  
```
	y_true = matches[i,valid]  - 正例样本
	y_score = -distmat[i][indices[i]][valid] - measure decision,取负
	average(average_precision_score(y_true, y_score))
```

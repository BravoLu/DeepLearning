# Code Reference
## evaluation_metrics/ 
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
## datasets/
将数据集统一成一样的格式。

## feature_extraction/ 
### database.py
FeatureDatabase继承Torch.utils.data中的Dataset.
### cnn.py
modules：这个参数的作用？？？
```
def extract_cnn_feature(model, inputs, modules=None):
	if modules is None:
		return model(inputs).data


```

## loss/
### oim.py
```
# 继承autograd.Function
# Function由forward,backward的方法
class OMI(autograd.Function):


class OIMLoss(nn.Module):
	
```
### triplet.py
```

```
## models/
num_features - 是否将特征提取设置为指定维度
### resnet.py
```
	ResNet(depth, pretrained=True,
		cur_at_pooling=False,num_features=0,
		norm=False,dropout=0,num_classes=0)
	cur_at_pooling - 如果True，将取出最后一个global pooling层前的model，并ignore剩下的参数

	num_features   - 如果为正，则在global pooling层后面加一个全连接层，输出结点数为num_features。后面
	加个BN层。Default:256 for 'inception' 0 for resnet

	norm           - True:将feature归一化为L2-norm，否者会加一个ReLU层
	dropout        - 

	num_classes    - 如果为正，会在尾端加一个全连接层作为分类器




	pretrained=False执行reset_params()。初始化参数
	self.modules(): 返回网络结构
```

### inception.py
```
	InceptionNet(cut_at_pooling=False,num_features=256,norm=False,dropout=0,num_classes=0)
```

## metric_learning/

## utils/data/
### dataset.py
```
	Data(root, split_id=0)

	split_id - 从第几个split_id开始

	Data.load(num_val=0.3,verbose=True) - num_val 测试集的比率

	_pluck(identities,indices,relabel=False) - relabel对不同的Train,val集重新设置label.
	根据indices从identities中获得信息，并返回(filename,person_id,camera_id)


```
### preprocessor.py
```
	Proprocessor(dataset, root=None, transform=None)
	_get_single_item(self,index):	返回(img,filename,person_id,camera_id)

```

### sampler.py
```
	RandomIdentitySample(data_source, num_instances=1)
	# 应该是随机取样
```

### transforms.py
```
	RectScale(height, width, interpolation=Image.BILINEAR):
		w,h = img.size
		return img.resize((self.width,self.height),self.interpolation)

	RandomSizedRectCrop(height,width,interpolation=Image.BILINEAR) - 先从图片中随机取一部分，然后再将它resize成相应大小

```
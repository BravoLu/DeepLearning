# open-reid
## Data Modules
### Unified Data Format
```
cuhk03
├── raw/
├── images/
├── meta.json
└── splits.json
```
raw/: 原始数据
images/: '{:08d}_{:02d}_{:04d}.jpg'.format(person_id, camera_id, image_id)
meta.json: 
```
"identities": [
    [  # the first identity, person_id = 0
        [  # camera_id = 0
            "00000000_00_0000.jpg",
            "00000000_00_0001.jpg"
        ],
        [  # camera_id = 1
            "00000000_01_0000.jpg",
            "00000000_01_0001.jpg",
            "00000000_01_0002.jpg"
        ]
    ],
    [  # the second identity, person_id = 1
        [  # camera_id = 0
            "00000001_00_0000.jpg"
        ],
        [  # camera_id = 1
            "00000001_01_0000.jpg",
            "00000001_01_0001.jpg",
        ]
    ],
    ...
]
```
splits.json:
```
{
    "trainval": [0, 1, 3, ...],  # person_ids for training and validation
    "gallery": [2, 4, 5, ...],   # for test gallery, non-overlap with trainval
    "query": [2, 4, ...],        # for test query, a subset of gallery
}
``` 
### Data Loading System
![img](dataloading.png)

## Training Parameters
```
python examples/softmax_loss.py -d viper -b 64 -j 2 -a inception --logs-dir logs/softmax-loss/viper-inception
```
|parameter|meaning |e.g.|
|- | - | -| 
|-d| dataset name |cuhk03,cuhk01,market1501,dukemtmc,viper|
|-a| model architecture| resnet18,resnet34,resnet50, resnet101,inception| 
|-b| batch size| 4 GPUs with -b 256 will have 64 minibatch samples|
|--combine-trainval|先用不同的训练/测试集调参，然后组合起来训练最终的模型| -|
|--height,--width|image size| 256x128 for resnet* |
CUDA_VISIBLE_DEVICES=0,1,2,3 python 

## Resume from Checkpoint
每个EPOCH后，会保存一个checkpoint.pth.tar文件，从这个checkpoint开始用参数 
```
--resume /path/to/checkpoint.pth.tar
```
## Evaluate a Trained Model
评价模型时时，可以直接从最优参数开始。
```
--resume /path/to/model_best.pth.tar --evaluate
```
## Tips
初始化softmax分类器的权重为std=0.001的高斯分布。如果用pretrained的模型则LR要调大

### Benchmark
https://cysu.github.io/open-reid/examples/benchmarks.html

### SDK Level reference
```

	class reid.trainers.BaseTrainer(model,criterion)
	class reid.trainers.Trainer(model,criterion)

	class reid.evaluators.extract_feature(model,data_loader, print_freq=1, metric=None)
	class reid.evaluators.pairwise_distance(features,query=None,gallery=None,metric=None)
	class reid.evaluators.evaluate_all(distmat, query=None, gallery, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None, cmc_topk=(1,5,10))
	class reid.evaluators.Evaluator(model)

	class reid.dist_metric.DistanceMetric(algorithm='euclidean',*args,**kwargs)
```
### API Level reference
```
	reid.datasets.create(name,root,*args,**kwargs)
	name - dataset name
	root - path to the dataset directory
	split_id(int = 0) - The index of data split,default:0
	num_val(int or float = 100) int:number of validation identities.float:proportion of validation to all the trainval.
	download(bool=False) - True:download the dataset

	class reid.datasets.CUHK01(root,split_id=0,num_val=100,download=True)
	class reid.datasets.CUHK03(root,split_id=0,num_val=100,download=True)
	class reid.datatsets.DukeMTMC(root,split_id=0,num_val=100,download=True)
	class reid.datasets.Market1501(root,split_id=0,num_val=100,download=True)
	class reid.datasets.VIPeR(root,split_id=0,num_val=100,download=True) 

	reid.models.create(name,*arg,**kwargs)
	reid.models.inception(**kwargs) 
	reid.models.resnet18(**kwargs)
	reid.models.resnet34(**kwargs)
	reid.models.resnet50(**kwargs)
	reid.models.resnet101(**kwargs)
	reid.models.resnet152(**kwargs)
	class reid.models.InceptionNet(cur_at_pooling=False,num_features=256,norm=False,dropout=0,num_classes=0)
	class reid.models.ResNet(cur_at_pooling=False,num_features=256,norm=False,dropout=0,num_classes=0)
```

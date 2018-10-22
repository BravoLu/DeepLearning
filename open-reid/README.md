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

.. _cuhk03-benchmark:

^^^^^^
CUHK03
^^^^^^

   ========= ============ ======== ============ ========== ============== ===============
   Net       Loss         Mean AP  CMC allshots CMC cuhk03 CMC market1501 Training Script
   ========= ============ ======== ============ ========== ============== ===============
   Inception Triplet      N/A      N/A          N/A        N/A            N/A
   Inception Softmax      65.8     48.6         73.2       71.0           ``python examples/softmax_loss.py -d cuhk03 -a inception --combine-trainval --epochs 70 --logs-dir examples/logs/softmax-loss/cuhk03-inception``
   Inception OIM          71.4     56.0         77.7       76.5           ``python examples/oim_loss.py -d cuhk03 -a inception --combine-trainval --oim-scalar 20 --epochs 70 --logs-dir examples/logs/oim-loss/cuhk03-inception``
   ResNet-50 Triplet      **80.7** **67.9**     **84.3**   **85.0**       ``python examples/triplet_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/cuhk03-resnet50``
   ResNet-50 Softmax      62.7     44.6         70.8       69.0           ``python examples/softmax_loss.py -d cuhk03 -a resnet50 --combine-trainval --logs-dir examples/logs/softmax-loss/cuhk03-resnet50``
   ResNet-50 OIM          72.5     58.2         77.5       79.2           ``python examples/oim_loss.py -d cuhk03 -a resnet50 --combine-trainval --oim-scalar 30 --logs-dir examples/logs/oim-loss/cuhk03-resnet50``
   ========= ============ ======== ============ ========== ============== ===============

.. _market1501-benchmark:

^^^^^^^^^^
Market1501
^^^^^^^^^^

   ========= ============ ======== ============ ========== ============== ===============
   Net       Loss         Mean AP  CMC allshots CMC cuhk03 CMC market1501 Training Script
   ========= ============ ======== ============ ========== ============== ===============
   Inception Triplet      N/A      N/A          N/A        N/A            N/A
   Inception Softmax      51.8     26.8         57.1       75.8           ``python examples/softmax_loss.py -d market1501 -a inception --combine-trainval --epochs 70 --logs-dir examples/logs/softmax-loss/market1501-inception``
   Inception OIM          54.3     30.1         58.3       77.9           ``python examples/oim_loss.py -d market1501 -a inception --combine-trainval --oim-scalar 20 --epochs 70 --logs-dir examples/logs/oim-loss/market1501-inception``
   ResNet-50 Triplet      **67.9** **42.9**     **70.5**   **85.1**       ``python examples/triplet_loss.py -d market1501 -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/market1501-resnet50``
   ResNet-50 Softmax      59.8     35.5         62.8       81.4           ``python examples/softmax_loss.py -d market1501 -a resnet50 --combine-trainval --logs-dir examples/logs/softmax-loss/market1501-resnet50``
   ResNet-50 OIM          60.9     37.3         63.6       82.1           ``python examples/oim_loss.py -d market1501 -a resnet50 --combine-trainval --oim-scalar 20 --logs-dir examples/logs/oim-loss/market1501-resnet50``
   ========= ============ ======== ============ ========== ============== ===============

.. _dukemtmc-benchmark:

^^^^^^^^
DukeMTMC
^^^^^^^^

   ========= ============ ======== ============ ========== ============== ===============
   Net       Loss         Mean AP  CMC allshots CMC cuhk03 CMC market1501 Training Script
   ========= ============ ======== ============ ========== ============== ===============
   Inception Triplet      N/A      N/A          N/A        N/A            N/A
   Inception Softmax      34.0     17.4         39.2       54.4           ``python examples/softmax_loss.py -d dukemtmc -a inception --combine-trainval --epochs 70 --logs-dir examples/logs/softmax-loss/dukemtmc-inception``
   Inception OIM          40.6     22.4         45.3       61.7           ``python examples/oim_loss.py -d dukemtmc -a inception --combine-trainval --oim-scalar 30 --epochs 70 --logs-dir examples/logs/oim-loss/dukemtmc-inception``
   ResNet-50 Triplet      **54.6** **34.6**     **57.5**   **73.1**       ``python examples/triplet_loss.py -d dukemtmc -a resnet50 --combine-trainval --logs-dir examples/logs/triplet-loss/dukemtmc-resnet50``
   ResNet-50 Softmax      40.7     23.7         44.3       62.5           ``python examples/softmax_loss.py -d dukemtmc -a resnet50 --combine-trainval --logs-dir examples/logs/softmax-loss/dukemtmc-resnet50``
   ResNet-50 OIM          47.4     29.2         50.4       68.1           ``python examples/oim_loss.py -d dukemtmc -a resnet50 --combine-trainval --oim-scalar 30 --logs-dir examples/logs/oim-loss/dukemtmc-resnet50``
   ========= ============ ======== ============ ========== ============== ===============

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

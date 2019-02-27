# Machine Learning concepts
This doc marks down the Deep Learning(Machine Learning) concepts that I never heard of. 

## end-to-end learning
非端到端：传统的机器学习的流程往往由多个模块组成，比如一个典型的NLP，包括分词，词性标注等多个独立步骤，每个步骤是一个独立的任务，其结果的好坏会影响到下一步骤，从而影响整个训练的结果

端到端：深度学习都是端对端学习


## receptive field
感受野：

## hinge function/loss

## train from scratch
从头开始训练

## [b, h , w , c]
b: batchsize
h: height
w: with
c: channels

## upsampling and downsampling

## Loss function & cost function
cost function 是 loss function在所有样本点上的总和。
经验风险损失函数和结构风险损失函数：经验风险损失函数指预测结果和实际结果的差别，结构风险损失函数是在经验风险损失函数加上正则项。


## Cross-Entropy
KL散度（相对熵）：In the context of machine learning,D_{kl}(P||Q) is offen called the information gain achieved if P is used instead of Q
(https://blog.csdn.net/tsyccnh/article/details/79163834)

## ROC & AUC
ROC (FPR)x-axis (TPR)y-axis
Advantage: ROC不随阈值改变而改变。 

## 偏差和方差
generalization error = bias + variance + noise
偏差大说明模型欠拟合，方差大说明模型过拟合。
![img](img/readme/bias&variance.png)

## cost curve
代价曲线：
在均等代价时，ROC曲线不能直接反应出模型的期望总体代价，而代价曲线可以。
代价曲线横轴为[0,1]的正例函数代价：

<img src="http://latex.codecogs.com/gif.latex?P(+)Cost=\frac{p\*Cost_{01}}{p\*Cost_{01}+(1-p)\*Cost_{10}}" />


其中p是样本为正例的概率。

代价曲线纵轴维[0,1]的归一化代价：
<img src="http://latex.codecogs.com/gif.latex?Cost_{norm}=\frac{FNR*p*Cost_{01}+FNR*(1-p)*Cost_{10}}{p*Cost_{01}+(1-p)*Cost_{10}}" />

## （参数估计）点估计，区间估计，中心极限定理之间的联系
点估计：用样本统计量来估计总体参数，因为样本统计量为数组上某一点指，估计的结果也以一个点的数值表示，所以称为点估计。
区间估计：通过从总体中抽取的样本，根据一定的正确度与精确度的要求，构造出适当的区间，以作为总体的分布参数（或参数的函数）的真值所在范围的估计。
1. 中心极限定理是推断统计的理论基础，推断统计包括参数估计和假设检验，其中参数估计包括点估计和区间估计，所以说，中心极限定理也是点估计和区间估计的理论基础。
2. 参数估计有两种方法：点估计和区间估计，区间估计包括了点估计。
相同点：都是基于一个样本作出；
不同点：点估计只提供单一的估计值，而区间估计基于点估计还提供误差界限，给出了置信区间，受置信度的影响。


## 类别不平衡问题解决方法
1. 扩大数据集
2. 对大类数据欠采样
3. 对小类数据过采样
4. 使用新评价指标
5. 选择新算法
6. 数据代价加权（例如当分类任务是识别小类，可以对分类器的小类样本数据增加权值，降低大类样本的权值）

## Kernel function 
将原坐标系里线性不可分的数据用Kernel投影到另一个空间，尽量使得数据在新的空间里线性可分


## 神经网络常用模型
![img](img/readme/neuralnetwork.jpg)

## 归一化
1. 为了后面数据处理的方便，归一化的确可以避免一些不必要的数值问题
2. 加快收敛
3. 同一量纲。样本数据的评价标准不一样，需要对其量纲化，统一评价标准，这是应用层面的需求。
4. 避免神经元饱和 —— Batch Normalization
5. 保证输出数据中数值小的不被吞食。

## LRN- 局部响应归一化
ALexNet中提出，后来证实没有用。

## BN算法的优点
1. 减少人为选择参数。某些情况下可以取消dropout和L2正则项参数，或者采用更小的L2正则项约束参数。
2. 减少了对学习率的要求。现在我们可以使用初始很大的学习率或者选择较小的学习率，算法也能够快速训练收敛。
3. 可以不再使用局部响应归一化。
4. 破坏原来的数据分布，一定程度上缓解过拟合（防止每批训练中某一样本经常被挑选到。）
5. 减少梯度消失，加快收敛速度，提高训练精度。

## finetuning
用别人的参数，修改后的网络和自己的数据进行训练，使得参数适应自己的数据，这样一个过程，通常称之为微调。
三种状态：
1. 只预测，不训练
2. 只训练最后分类层
3. 完全训练，分类层+之前卷积层都训练。

## GN—— Group Normalization WN—— Weight Normalization
GN将通道分成组，并在每组内计算归一化的均值和方差
WN对网络权值W进行normalization.

## 无监督预训练
逐层贪婪训练，无监督训练即训练网络的第一个隐藏层，再训练第二个...，最后用这些训练好的参数值作为整体网络参数的初始值。

## 稀疏初始化

## Dropout
hidden layer dropout=0.5最好。

## BN与学习率的关系
增大batch-size相当于降低学习率

## multi-task learning

## Bottleneck

## softmax&cross entropy loss
entropy loss 如下：
![img](softmax.jpg)  
$p_j$为softmax得到的概率
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

$P(+)Cost=\frac{p\*Cost_{01}}{p\*Cost_{01}+(1-p)\*Cost_{10}}$

其中p是样本为正例的概率。

代价曲线纵轴维[0,1]的归一化代价：
$Cost_{norm}=\frac{FNR*p*Cost_{01}+FNR*(1-p)*Cost_{10}}{p*Cost_{01}+(1-p)*Cost_{10}}$
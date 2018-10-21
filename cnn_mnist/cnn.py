import torch
import torch.nn as nn
import torch.utils.data as Data 
import torchvision.datasets as dset
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import torch.optim as optim
import logging 

'''
	MNIST(root,train=True,transform=None,target_transform=None, download=False)
	root:root directory of dataset where processed/training.pt and processed/test.pt exist
	train:True - use training set,False - use test set.transform
	transform:transform to apply to input images
	target_transform: transform to apply to targets(class labels)
	download:whether to download the MNIST data
''' 
DOWNLOAD_MNIST = True
BATCH_SIZE = 50
LR = 0.001
EPOCH = 1


train_data = dset.MNIST(
	root='./mnist/',
	train=True,
	transform = transforms.ToTensor(),
	target_transform = None,
	download = DOWNLOAD_MNIST
)

test_data = dset.MNIST(
	root='./mnist/',
	train=False,
	transform = None,
	target_transform = None,
	download = DOWNLOAD_MNIST
)
'''
	DataLoader(dataset, batch_size=1, shuffle=False, sample=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
	dataset - 加载数据的数据集
	batch_size - 每个batch加载多少个样本
	shuffle - 设置为True时会在每个epoch重新打乱数据
	sampler - 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
	num_worker - 用多少个子进程加载数据。0表示数据将在主进程中加载
	drop_last - 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。
'''
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_x = torch.unsqueeze(test_data.test_data,dim=1).type(torch.FloatTensor)[:10000].cuda()/255
test_y = test_data.test_labels[:10000].cuda()


'''
	ReLU(inplace=False)  
	inplace-选择是否进行覆盖运算
	
	如何计算？
	BatchNorm2d(num_features,eps=1e-05,momentum=0.1,affine=True)
	num_features:来自期望输入的特征数，该期望输入的大小为'batch_size x num_features [x width]'
	eps:为保证数值稳定性（分母不能趋近或取0），给分母加上的值。默认为1e-5。
	momentum:动态均值和动态方差所使用的的动量，默认为0.1
	affine:一个布尔值，当设为true，给该层添加可学习的仿射变换参数。
	
	MaxPool2d(kernel_size, stride=None, padding=0, dialation=1, return_indices=False,ceil_mode=False)
	kernel_size - max pooling的窗口大小
	stride - max pooling的窗口移动的步长，默认值是kernel_size
	padding - 输入的每一条边补充0的层数
	dilation - 一个控制窗口中元素步幅的参数
	return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
	ceil_mode - 如果等于True,计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作


'''
def conv_layer(in_channels,out_channels,kernel,stride=1,padding=1):
	layer = nn.Sequential(
		nn.Conv2d(in_channels,out_channels,kernel,stride,padding),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2)
	)
	return layer


# net 
class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = conv_layer(1,16,5,1,2)
		#一层CNN
		self.FC = nn.Linear(16*14*14,10)


		#self.conv2 = conv_layer(16,32,5,1,2)
		#输入是32个7*7的矩阵
		#  nn.linear(in_features,out_features,bias=True)
		#self.FC = nn.Linear(32*7*7,10)

	def forward(self,x):
		x = self.conv1(x)
		#x = self.conv2(x)
		x = x.view(x.size(0), -1)
		output = self.FC(x)
		return output


# training
cnn = CNN()
cnn.cuda()
'''
	optimizer对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。
	构建：需要给它一个包含需要优化的参数（Variable对象）的iterable

	Optimizer也支持为每个参数单独设置选项，通过传入dict的iterable 
	e.g. 
		optim.SGD([
						{'params':model.base.parameters()},
						{'params':model.classifier.parameters(), 'lr':1e-3}],
						lr=1e-2, momentum=0.9)

		optimizer.step()

		zero_grad() : 清空所有被优化过的Variable的梯度			
'''

# 用SGD收敛慢很多?
#optimizer = optim.SGD(cnn.parameters(),lr=LR)
#optimizer = optim.RMSprop(cnn.parameters(),lr=LR)
#optimizer = optim.Adagrad(cnn.parameters(),lr=LR)
#optimizer = optim.momentum(cnn.parameters(),lr=LR)
optimizer = optim.Adam(cnn.parameters(),lr=LR)
plot_x = []
plot_y = []

loss_func  = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
	for step,(x,y) in enumerate(train_loader):
		b_x = x.cuda()
		b_y = y.cuda()

		plot_x.append(step)
		output = cnn(b_x)
		loss = loss_func(output,b_y)
		plot_y.append(loss.data.cpu().numpy())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		if step%50 == 0:
			test_output = cnn(test_x)
			# 返回所有张量中的最大值
			# .data 的用法
			'''
				squeeze : 将输出张量形状中的1去除并返回。
				squeeze(input, dim=None, out=None)
				(A x 1 x B x 1 x C x 1 x D) -> (A x B x C x D)
				当给定dim时，那么挤压操作只在给定维度上
				e.g. (A x 1 x B),只有squeeze(input,1)，形状会变成(A x B)
			'''

			#torch.max返回(val,index)
			pred_y = torch.max(test_output,1)[1].cuda().data.squeeze().numpy()
			acc = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
			#acc = float((pred_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
			#print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % acc)
			print('Epoch: {} | train loss: {:.4f} | test accuracy: {:.2f}'.format(epoch,loss.data.cpu().numpy(),acc))

torch.save(cnn,'cnn.pkl')
torch.save(cnn.state_dict(), 'cnn_params.pkl')


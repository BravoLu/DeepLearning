import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
import torch.optim as optim 
import torchvision

# model
class ResidualBlock(nn.Module):
	def __init__(self,in_channel,out_channel,stride=1):
		super(ResidualBlock,self).__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1,bias=False),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1,bias=False),
			nn.BatchNorm2d(out_channel)
		)
		self.shortcut = nn.Sequential()
		if stride != 1 or in_channel != out_channel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,bias=False),
				nn.BatchNorm2d(out_channel),
			)
	def forward(self,x):
		output = self.block(x)
		output += self.shortcut(x)
		output = F.relu(output)
		return output

class ResNet(nn.Module):
	def __init__(self, num_classes=10):
		super(ResNet,self).__init__()
		self.in_channel = 64 
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
		)
		self.conv2_x = self.make_layer(64,2,1)
		self.conv3_x = self.make_layer(128,2,2)
		self.conv4_x = self.make_layer(256,2,2)
		self.conv5_x = self.make_layer(512,2,2)
		self.FC      = nn.Linear(512, 10)

	def make_layer(self, out_channel, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(ResidualBlock(self.in_channel, out_channel, stride))
			self.in_channel = out_channel
		return nn.Sequential(*layers)

	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2_x(x)
		x = self.conv3_x(x)
		x = self.conv4_x(x)
		x = self.conv5_x(x)
		x = F.avg_pool2d(x,4)
		x = x.view(x.size(0),-1)
		x = self.FC(x)
		return x

# parameter setting
EPOCH = 100
BATCH_SIZE = 128
LR = 0.1

transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),  
   		transforms.RandomHorizontalFlip(),  
		transforms.ToTensor(),
		transforms.Normalize((0.4914,0.4822,0.4465), (0.2023, 0.1994, 0.2010)),
	])
transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914,0.4822,0.4465), (0.2023, 0.1994, 0.2010)),
	])

# data preprocessing
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='./data',train=False,download=False, transform=transform_test) 

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = ResNet()
if torch.cuda.is_available():
	model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR) #momentum, weight_decay

#Training 
for epoch in range(EPOCH):
	model.train()
	for step,(x,y) in enumerate(train_loader):
		print(step,end='\r')
		if torch.cuda.is_available():
			x = x.cuda()
			y = y.cuda()
		optimizer.zero_grad()

		output = model(x)
		loss = criterion(output, y)
		loss.backward()
		optimizer.step()

	
	with torch.no_grad():
		model.eval()		
		losses = 0.0
		acc_nums = 0
		num_batch = 0
		num_instance = 0

		for step,(x,y) in enumerate(test_loader):
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
			output = model(x)
			loss = criterion(output,y) 
			pred_y = torch.max(output.data, 1)[1].data.squeeze()
			acc = torch.sum(pred_y==y.data.squeeze())
			
			losses += loss.item()
			acc_nums += acc.item()
			num_batch += 1
			num_instance += pred_y.size(0)

		loss = losses / num_batch
		acc = acc_nums / num_instance
		
		print('EPOCH: {} | Test loss: {:.4f} | Test Acc {:.2f}'.format(epoch,loss,acc))

torch.save(cnn.state_dict(), 'ResNet_params.pth')



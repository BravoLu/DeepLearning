# -*- coding: utf-8 -*-
# @Author: Lu Shaohao(Bravo)
# @Date:   2018-11-26 10:19:51
# @Last Modified by:   Lu Shaohao(Bravo)
# @Last Modified time: 2018-11-28 10:30:15

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms 
import torch.optim as optim 
import torchvision

#DenseNet-121

class DenseBlock(nn.Module):
	def __init__(self, in_channel, out_channel, drop=0):
		tmp_channel = out_channel 
		self.conv1 =  nn.conv2d(in_channel, tmp_channel, kernel_size=1, padding=0, stride=1, bias=False)
		self.bn1   =  nn.BatchNorm2d(in_channel)
		self.relu =  nn.ReLU(inplace=True)

		self.conv2 =  nn.conv2d(tmp_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
		self.bn2   =  nn.BatchNorm2d(tmp_channel)	

	def forward(self,x):
		out = self.bn1(x)
		out = self.relu(out) 
		out = self.conv1(out) 
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv2(out)
		return torch.cat([x,out],1)

class DenseLayer(nn.Module):
	def __init__(self,in_channel,out_channel, growth_rate=8  ,drop=0):
		self.layer = self._make_layer()
	def _make_layer(self, in_channel, growth_rate, nb_layers, drop=0):
		layers = []
		for i in range(nb_layers):
			layers.append(DenseBlock(in_channel+i*growth_rate, in_channel+(i+1)*growth_rate, drop))
		return nn.Sequential(*layers)
	def forward(self,x):
		return self.layer(x)

class TransitionLayer(nn.Module):
	def __init__(self,in_channel,out_channel):
		self.bn1   = nn.BatchNorm2d(in_channel)
		self.relu  = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
	def forward(self,x):
		x = self.conv1(self.relu(self.bn1(x)))
		out = F.avg_pool2d(out,kernel_size=2,stride=2)

class DenseNet(nn.Module):
	def __init__(self, nb_denseblock_per_layer, num_classes=10, growth_rate=12 , reduction=0.5):
		super(DenseNet,self).__init__()

		self.conv1 = nn.Sequential(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.mp    = nn.MaxPool2d(kernel_size=3, stride=2)
		self.in_channel = int(64 + nb_denseblock_per_layer * growth_rate)
		self.denseblock1 = DenseLayer(64, in_channel, growth_rate)
		self.trans1      = TransitionLayer(in_channel, int(math.floor(in_channel*reduction)))
		self.in_channel = int(math.floor(in_channel*reduction))
		self.out_channel = int(in_channel + nb_denselock_per_layer * growth_rate)
		self.denseblock2 = DenseLayer(in_channel, out_channel ,growth_rate)
		self.trans2      = TransitionLayer(out_channel, int(math.floor(out_channel*reduction)))
		self.fc          = nn.Linear(int(math.floor))      
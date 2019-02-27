from __future__ import absolute_import
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import init
import math


# kernelsize=3, stride=1, padding=1
def conv_3x3_bn(in_planes, out_planes, kernel_size=3, stride, padding=1):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size , stride, padding, bias=False),
		nn.BatchNorm2d(out_planes),
		nn.ReLU6(inplace=True)
	)

# kernelsize=1, stride=1, padding=0
def conv_1x1_bn(in_planes, out_planes, kernel_size=1, stride=1, padding=0):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
		nn.BatchNorm2d(out_planes),
		nn.ReLU6(inplace=True)
	)

def depthwise_conv(in_planes, out_planes, kernel_size=3, stride, padding=1, bias=False):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=in_planes, bias),
		nn.BatchNorm2d(out_planes),
		nn.ReLU6(inplace=True)
	)

def pointwise_conv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias),
		nn.BatchNorm2d(out_planes),
		nn.ReLU6(inplace=True)
	)

def pointwise_linear_conv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)):
	return nn.Sequential(
		nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias),
		nn.BatchNorm2d(out_planes)
	)	

class InvertedResidual(nn.Module):
	def __init__(self, in_planes, out_planes, stride, expand_ratio):
		super(InvertedResidual,self).__init__()
		self.stride = stride
		assert stride in [1,2]

		hidden_planes = round(in_planes * expand_ratio)
		self.use_res_connect = self.stride == 1 and in_planes == out_planes

		if expand_ratio == 1:
			self.conv = nn.Sequential(
				depthwise_conv(hidden_planes,hidden_planes,stride=stride),
				pointwise_linear_conv(hidden_planes, out_planes)
			)
		else:
			self.conv = nn.Sequential(
				pointwise_conv(in_planes, hidden_planes),
				depthwise_conv(hidden_planes, hidden_planes, stride=stride),
				pointwise_linear_conv(hidden_planes, out_planes)
			)

	def forward(self,x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)

class MobileNetV2(nn.Module):
	def __init__(self, pretrained=True, cut_at_pooling=False, num_features=256, norm=False, dropout=0, num_classes=0):
		super(MobileNetV2,self).__init__()
		input_size = 224
		width_mult = 1.
		input_channel = 32
		last_channel = 1280
		param_setting = [
			# t, c, n, s
			[1, 16, 1, 1],
			[6, 24, 2, 2],
			[6, 32, 3, 2],
			[6, 64, 4, 2],
			[6, 96, 3, 1],
			[6, 160,3, 2],
			[6, 320,1, 1],
		]

		assert input_size % 32 == 0 
		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

		#first layer
		self.features = [conv_3x3_bn(3, input_channel, stride=2)]
		for t,c,n,s in param_setting:
			output_channel = int(c*width_mult)
			for i in range(n):
				if i == 0:
					self.features.append(InvertedResidual(input_channel, output_channel, stride=s, expand_ratio=t))
				else:
					self.features.append(InvertedResidual(input_channel, output_channel, stride=1, expand_ratio=t))
				input_channel = output_channel

		self.features.append(conv_1x1_bn(input_channel,self.last_channel))
		self.features = nn.Sequential(*self.features)

		if not self.cut_at_pooling:
			self.num_features = num_features
			self.norm = norm
			self.dropout = dropout
			self.has_embedding = num_features > 0
			self.num_classes = num_classes

			if self.has_embedding:
				self.feat = nn.Linear(self.last_channel,self.num_features)
				self.feat_bn = nn.BatchNorm1d(self.num_features)
			else:
				self.num_features = self.last_channel

			if self.dropout > 0:
				self.drop = nn.Dropout(self.dropout)

			if self.num_classes > 0:
				self.classifier = nn.Linear(self.num_features, self.num_classes)

		self._initialize_weights()

	def forward(self,x):
		x = self.features(x)

		if self.cut_at_pooling:
			return x

		x = x.mean(3).mean(2)

		if self.has_embedding:
			x = self.feat(x)
			x = self.feat_bn(x)
		if self.norm:
			x = F.normalize(x)
		elif self.has_embedding:
			x = F.relu(x)
		if self.dropout > 0:
			x = self.drop(x)
		if self.num_calsses > 0:
			x = self.classifier(x)
			
		return x


	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0,0.01)
				m.bias.data.zero_()



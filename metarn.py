import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class Naive(nn.Module):

	def __init__(self):
		super(Naive, self).__init__()

		self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3),
		                         nn.MaxPool2d(kernel_size=2),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.MaxPool2d(kernel_size=2),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),

		                         nn.Conv2d(64, 64, kernel_size=3),
		                         nn.BatchNorm2d(64),
		                         nn.ReLU(inplace=True),
		                         )
		# Avg Pooling is better
		self.downsample = nn.Sequential(nn.AvgPool2d(5,5))

	def forward(self, input):
		input =  self.net(input)
		return self.downsample(input)

class RN(nn.Module):

	def __init__(self, c, d):
		super(RN, self).__init__()

		# the input is self.c with two coordination information, and then combine each
		self.g = nn.Sequential(nn.Linear( (self.c + 2) * 2, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.ReLU(inplace=True))

		self.f = nn.Sequential(nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.Dropout(),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 64),
		                       nn.BatchNorm1d(64),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(64, 1),
		                       nn.Sigmoid())

	def forward(self, input):
		batchsz, c, d, _ = input.size()
		# input [batch, c, d, d] => [b, c, d*d] => [b, c, 1, d*d] => [b, c, d*d, d*d]
		x_i = input.view(batchsz, c, d * d).unsqueeze(2).expand(batchsz, c, d * d, d * d)
		# input [batch, c, d, d] => [b, c, d*d] => [b, c, d*d, 1] => [b, c, d*d, d*d]
		x_j = input.view(batchsz, c, d * d).unsqueeze(3).expand(batchsz, c, d * d, d * d)
		# [b, 2c, d*d, d*d] => [b, d*d, d*d, 2c] => [b * d*d * d*d, 2c]
		x = torch.cat([x_i, x_j], dim = 1).transpose(1, 3).contiguous().view(batchsz * d*d * d*d, 2 * c)
		# push to G network
		# [b*d^4, 2c] => [b*d^4, -1] => [b, d^4, -1] => [b, -1]
		x = self.g(x).view(batchsz, d**4, -1).sum(1)
		# [b, -1] => [b, 1]
		x = self.f(x)

		return x

class MetaRN(nn.Module):

	def __init__(self):
		super(MetaRN, self).__init__()

		self.repnet = Naive()
		self.rn = RN()



	def forward(self, support_x, support_y, query_x, query_y):
		"""
		query_x is current learning obj,
		support_x is previous learned obj
		:param support_x: [b, setsz, c_, h, w]
		:param support_y:
		:param query_x:   [b, querysz, c_, h, w]
		:param query_y:
		:return:
		"""
		# [b, setsz/querysz, c_, h, w] => [b, setsz+querysz, c_, h, w]
		input = torch.cat([support_x, query_x], dim = 1)
		# [batchsz, c_, h, w] => [b, c, d, d]
		x = self.repnet(input)




























































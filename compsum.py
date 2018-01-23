import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from repnet import repnet_deep, Bottleneck


class CompSum(nn.Module):
	"""
	Treat 5 way images as a big images and compare between objects and then relation predict.
	Compare and then sum over G and then push to F
	repnet => feature concat => layer4 & layer5 & avg pooling => fc => sigmoid
	"""
	def __init__(self, n_way, k_shot):
		super(CompSum, self).__init__()

		self.n_way = n_way
		self.k_shot = k_shot
		assert  k_shot == 1

		self.repnet = repnet_deep(False)
		# we need to know the feature dim, so here is a forwarding.
		repnet_output = self.repnet(Variable(torch.rand(2, 3, 224, 224)))
		repnet_sz = repnet_output.size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		# this is the input channels of layer4&layer5
		self.inplanes = 2 * self.c
		assert repnet_sz[2] == repnet_sz[3]
		print('repnet sz:', repnet_sz)

		# after relational module
		# concat 2* (1024, 14, 14) => [256, 4, 4] => [256], pooling
		self.layer4 = self._make_layer(Bottleneck, 128, 4, stride=2)
		self.layer5 = self._make_layer(Bottleneck, 64, 3, stride=2)
		self.g = nn.Sequential(self.layer4,
		                       self.layer5,
		                       nn.AvgPool2d(4))
		self.f = nn.Sequential(
			nn.Linear(256 , 64),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.Linear(64, n_way)
		)

		self.criteon = nn.CrossEntropyLoss()

	def _make_layer(self, block, planes, blocks, stride=1):
		"""
		make Bottleneck layer * blocks.
		:param block:
		:param planes:
		:param blocks:
		:param stride:
		:return:
		"""
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)


	def forward(self, support_x, support_y, query_x, query_y, train=True):
		"""
		query_y is the index of support_y, not real label.
		:param support_x: [b, setsz, c_, h, w]
		:param support_y: [b, setsz]
		:param query_x:   [b, querysz, c_, h, w]
		:param query_y:   [b, querysz]
		:return:
		"""
		batchsz, setsz, c_, h, w = support_x.size()
		querysz = query_x.size(1)
		c, d = self.c, self.d

		# [b, setsz, c_, h, w] => [b*setsz, c_, h, w] => [b*setsz, c, d, d] => [b, setsz, c, d, d]
		support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
		# [b, querysz, c_, h, w] => [b*querysz, c_, h, w] => [b*querysz, c, d, d] => [b, querysz, c, d, d]
		query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

		# concat each query_x with all setsz along dim = c
		# [b, setsz, c, d, d] => [b, 1, setsz, c, d, d] => [b, querysz, setsz, c, d, d]
		support_xf = support_xf.unsqueeze(1).expand(-1, querysz, -1, -1, -1, -1)
		# [b, querysz, c, d, d] => [b, querysz, 1, c, d, d] => [b, querysz, setsz, c, d, d]
		query_xf = query_xf.unsqueeze(2).expand(-1, -1, setsz, -1, -1, -1)
		# cat: [b, querysz, setsz, c, d, d] => [b, querysz, setsz, 2c, d, d]
		comb = torch.cat([support_xf, query_xf], dim=3)

		# push G network
		# [b*querysz*setsz, 2c, d, d] => [b*querysz*setsz, 256] => [b, querysz, setsz, 256]
		# [10, 5, 5, 4096]
		comb = self.g(comb.view(batchsz * querysz * setsz, 2 * c, d, d)).view(batchsz, querysz, setsz, -1)
		# [b, querysz, setsz, -1] => [b, querysz, -1]
		comb = comb.sum(dim = 2)
		# push to Linear layer
		# [b * querysz, 256] => [b * querysz, n_way] => [b, querysz, n_way]
		score = self.f(comb.view(batchsz * querysz, -1)).view(batchsz, querysz, self.n_way)


		# score: [b, querysz, n_way]
		# label: [b, querysz]
		# [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
		support_y_ = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
		# [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
		query_y_ = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
		# [b, querysz, setsz] => [b*querysz, 3], while b*querysz is number of non-zero, and 3 is the dim of tensor
		query_y_idx = torch.eq(support_y_, query_y_).nonzero()
		# only retain the last index, => [b, querysz]
		query_y_idx = (query_y_idx[...,-1]).contiguous().view(batchsz, querysz)
		if train:
			loss = self.criteon(score.view(batchsz * querysz, self.n_way), query_y_idx.view(-1))
			return loss

		else:
			# [b, querysz, n_way] => [b, querysz, n_way]
			indices = F.softmax(score, dim= 2)
			# [b, querysz, n_way] => [b, querysz], indx
			_, indices = torch.max(indices, dim = 2)
			# [b, setsz] along with dim = 1, index = softmax output
			# => [b, querysz]
			pred = torch.gather(support_y, dim= 1, index= indices)

			correct = torch.eq(pred, query_y).sum()
			return pred, correct

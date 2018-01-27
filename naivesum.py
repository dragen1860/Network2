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


class NaiveSum(nn.Module):
	"""
	Ensemble features after repnet and push to G & F network. the n-way output of F will be concated and then
	push to O network.
	"""
	def __init__(self, n_way, k_shot, imgsz):
		super(NaiveSum, self).__init__()

		self.n_way = n_way
		self.k_shot = k_shot

		self.repnet = Naive()

		# we need to know the feature dim, so here is a forwarding.
		# => [64, 10, 10]
		repnet_sz = self.repnet(Variable(torch.rand(2, 3, imgsz, imgsz))).size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		assert repnet_sz[2] == repnet_sz[3]
		print('repnet sz:', repnet_sz)

		# the input is self.c with two coordination information, and then combine each
		# 2c+4 => 256
		self.g = nn.Sequential(nn.Linear( (self.c + 2) * 2, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.BatchNorm1d(256),
		                       nn.ReLU(inplace=True))

		# 256 => 64
		rn_dim = 64 # output dim of F network
		self.f = nn.Sequential(nn.Linear(256, 128),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(128, 128),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(128, rn_dim),
		                       nn.Dropout(),
		                       nn.ReLU(inplace=True))

		# 256 => n-way
		# we will cat output features of F network along n-way
		self.o = nn.Sequential(nn.Linear(n_way * rn_dim, 64),
		                       nn.BatchNorm1d(64),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(64, n_way))

		self.criteon = nn.CrossEntropyLoss()

		coord = np.array([(i / self.d , j / self.d) for i in range(self.d) for j in range(self.d)])
		self.coord = torch.from_numpy(coord).float().view(self.d, self.d, 2).transpose(0, 2).transpose(1,2).contiguous()
		self.coord = self.coord.unsqueeze(0).unsqueeze(0)
		# print('self.coord:', self.coord.size(),self.coord) # [batchsz:1, setsz:1, 2, self.d, self.d]



	def forward(self, support_x, support_y, query_x, query_y, train = True):
		"""

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

		# sum over k_shot imgs to ensemble
		# [b, setsz, c, d, d] => [b, n_way, k_shot, c, d, d], sum over k_shot dim
		# => [b, n_way, c, d, d]
		support_xf = support_xf.view(batchsz, self.n_way, self.k_shot, c, d, d).sum(2)
		# update setsz now
		setsz = self.n_way
		# [b, n_way*k_shot] => [b, n_way]
		support_y = support_y[:, ::self.k_shot]


		## now make the combination between two pairs
		# include the coordinate information in each feature dim
		# [b, setsz/querysz, c+2, d, d]
		support_xf = torch.cat([support_xf, Variable(self.coord.expand(batchsz, setsz, 2, d, d)).cuda()], dim = 2)
		query_xf = torch.cat([query_xf, Variable(self.coord.expand(batchsz, querysz, 2, d, d)).cuda()], dim = 2)
		c += 2 # c is a copy of self.c, we need not reset c since it will be reseted in the beginning of forward

		# make combination now
		# [b, setsz, c, d, d] => [b, setsz, c, d*d] => [b, 1, setsz, c, d*d] => [b, 1, setsz, c, 1, d*d] => [b, querysz, setsz, c, d*d, d*d]
		support_xf = support_xf.view(batchsz, setsz, c, d*d).unsqueeze(1).unsqueeze(4).expand(batchsz, querysz, setsz, c, d*d, d*d)
		# [b, querysz, c, d, d] => [b, querysz, c, d*d] => [b, querysz, 1, c, d*d, 1] => [b, querysz, setsz, c, d*d, d*d]
		query_xf = query_xf.view(batchsz, querysz, c, d*d).unsqueeze(2).unsqueeze(5).expand(batchsz, querysz, setsz, c, d*d, d*d)
		# [b, querysz, setsz, c*2, d*d, d*d]
		comb = torch.cat([support_xf, query_xf], dim=3)

		# [b, querysz, setsz, c*2, d*d:0, d*d:1] => [b, querysz, setsz, d*d:1, d*d:0, c*2] => [b, querysz, setsz, d*d:0, d*d:1, c*2]
		comb = comb.transpose(3, 5).transpose(3, 4).contiguous().view(batchsz * querysz * setsz * d*d * d*d, c * 2)
		# push to G network
		# [b*querysz*setsz*d^4, 2c] => [b*querysz*setsz*d^4, -1]
		x_f = self.g(comb)
		# sum over coordinate axis and squeeze it
		# [b*querysz*setsz*d^4, -1] => [b*querysz*setsz*d^4, -1] => [b*querysz*setsz, d^4, 64] => [b*querysz*setsz, -1]
		x_f = x_f.view(batchsz * querysz * setsz, d*d * d*d, -1).sum(1) # the last dim can be derived by layer setting
		# push to F network
		# [batchsz * querysz * setsz, -1] => [batchsz * querysz * setsz, -1] => [b, querysz, setsz, -1]
		rn = self.f(x_f).view(batchsz, querysz, setsz, -1)

		# push to O network
		# [b, querysz, setsz, -1] => [b * querysz, setsz * dim_f_out]
		# cat along feature/rn dim
		rn = rn.view(batchsz * querysz, -1)
		# [b*querysz, setsz*dim_f_out] => [b*querysz, prob] => [b, querysz, prob]
		logits = self.o(rn).view(batchsz, querysz, self.n_way)

		# build its label
		# logits: [b, querysz, n_way]
		# [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
		support_y_ = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
		# [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
		query_y_ = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
		# [b, querysz, setsz] => [b*querysz, 3], while b*querysz is number of non-zero, and 3 is the dim of tensor
		query_y_idx = torch.eq(support_y_, query_y_).nonzero()
		# only retain the last index, => [b, querysz]
		query_y_idx = (query_y_idx[...,-1]).contiguous().view(batchsz, querysz)
		# print(support_y[0].cpu().data.numpy())
		# print(query_y[0].cpu().data.numpy())
		# print(query_y_idx[0].cpu().data.numpy())



		# logits: [b, querysz, prob/n-way]
		# query_y_idx: [b, querysz]
		# query_y: [b, querysz]
		# NOTICE: since use just the global true label for support_y and query_y, not relative label from 0 to n-way,
		# we need to convert global true label to relative index
		# for instance: support_y: [9..., 8..., 12..., 33..., 63...] query_y: [33, 63, 9, 12, 8]
		# we get relative index: query_y_idx: [3, 4, 0, 2, 2]
		if train:

			# APPROACH 1: euclidean loss
			# [b, querysz, nway] => [b, querysz, n_way]
			prob = F.sigmoid(logits)
			# [b, querysz, n_way] scatter with [b, querysz, 1] => one hot [b, querysz, n-way]
			query_y_idx_onehot = torch.zeros_like(prob).scatter_(2, query_y_idx.unsqueeze(2), 1).float()
			# print('onehost', label[0].cpu().data.numpy())
			# loss = self.criteon(score.view(batchsz * querysz, self.n_way), query_y_idx.view(-1))
			loss = torch.pow(prob - query_y_idx_onehot, 2).sum() / batchsz
			return loss


			# # APPROACH 2: cross entropy loss
			# # logits: [b, querysz, n-way]
			# # query_y_idx: [b, querysz]
			# loss = self.criteon(logits.view(batchsz * querysz, self.n_way), query_y_idx.view(-1)) * batchsz
			# return loss

		else:
			# [b, querysz, n-way] => [b, querysz]
			_, indices = logits.max(dim=2)
			# support_y: [b, querysz]
			# indices: [b, querysz]
			# pred: [b, querysz], global true label
			pred = torch.gather(support_y, dim=1, index=indices)

			correct = torch.eq(pred, query_y).sum()
			return pred, correct
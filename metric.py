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


class Metric(nn.Module):
	"""
	Different from naivesum, which do more complex ensemble work. here we just sum over features after repnet.
	"""
	def __init__(self, n_way, k_shot, imgsz):
		super(Metric, self).__init__()

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
		                       nn.ReLU(inplace=True))

		# output distance between two pairs.
		self.f = nn.Sequential(nn.BatchNorm1d(256),
								nn.Linear(256, 256),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 256),
		                       nn.Dropout(),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(256, 64),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(64, 1),
		                       nn.Sigmoid())


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
		:param train:
		:return:
		"""
		batchsz, setsz, c_, h, w = support_x.size()
		querysz = query_x.size(1)
		c, d = self.c, self.d

		# [b, setsz, c_, h, w] => [b*setsz, c_, h, w] => [b*setsz, c, d, d] => [b, setsz, c, d, d]
		support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
		# [b, querysz, c_, h, w] => [b*querysz, c_, h, w] => [b*querysz, c, d, d] => [b, querysz, c, d, d]
		query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

		# # sum over k_shot imgs' features to ensemble
		# # [b, setsz, c, d, d] => [b, n_way, k_shot, c, d, d] => [b, n_way, c, d, d], sum over k_shot dim
		# support_xf = support_xf.view(batchsz, self.n_way, self.k_shot, c, d, d).sum(2)
		# # update setsz now
		# setsz = self.n_way
		# # [b, n_way*k_shot] => [b, n_way]
		# support_y = support_y[:, ::self.k_shot]


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
		comb = comb.transpose(3, 5).contiguous().view(batchsz * querysz * setsz * d*d * d*d, c * 2)
		# push to G network
		# [b*querysz*setsz*d^4, 2c] => [b*querysz*setsz*d^4, -1/256]
		x_f = self.g(comb)
		# sum over coordinate axis and squeeze it
		# [b*querysz*setsz*d^4, -1] => [b*querysz*setsz*d^4, -1] => [b*querysz*setsz, d^4, -1] => [b*querysz*setsz, -1]
		x_f = x_f.view(batchsz * querysz * setsz, d*d * d*d, -1).sum(1) # the last dim can be derived by layer setting
		# push to F network
		# [batchsz * querysz * setsz, -1] => [batchsz * querysz * setsz, 1] => [b, querysz, setsz]
		dist = 1 - self.f(x_f).view(batchsz, querysz, setsz)

		# build its label
		# [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
		support_yf = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
		# [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
		query_yf = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
		# eq: [b, querysz, setsz] => [b, querysz, setsz] and convert byte tensor to float tensor
		# this is the label of distance, where 0 means positive pairs
		label = torch.ne(support_yf, query_yf).float()
		# similarity label, for the following select dist(a,p) usage.
		label_sim = torch.eq(support_yf, query_yf)



		# here we do NOT use similarity score, in which 1 means similar pairs. Instead we use
		# distance metric, where 0 means the closet and 1 means negative pairs.
		# dist: [b, querysz, setsz]
		# label: [b, querysz, setsz]
		if train:
			# construct triplet loss.
			# loss = sum{ max(dist(a,p) - min_neg(dist(a,n)) + margin, 0) }
			margin = 0.2
			# min_n(dist(a,n))
			# [b, querysz, setsz] * [b, querysz, setsz]
			# global label: [1, 2, 3, 4, 5], current anchor label: 3
			# label:        [1, 1, 0, 1, 1]
			# dist:         [0.4, 0.3, 0.2, 0.8, 0.9] * [1, 1, 0, 1, 1] => [0.4, 0.3, 0, 0.8, 0.9]
			# select the 2nd small element, since it's min_n(dist(a,n))

			# # [b, querysz, setsz] => [b, querysz]
			# min_neg, _ = torch.kthvalue(torch.mul(label, dist).cpu(), 2)
			# min_neg = min_neg.cuda()

			# sort [b, querysz, setsz] => [b, querysz, setsz]
			min_neg, _ = torch.sort(torch.mul(label, dist), dim=2)
			# [b, querysz, setsz] => [b, querysz], get the 2nd small element
			min_neg = min_neg[:, :, 1]

			# now select dist(a,p)
			# torch.where function exists only in torch v0.4, shit!
			# [b, querysz, setsz] => [N, 3], the last column is positive index
			indices = label_sim.nonzero()
			# [N, 3] => [N, 1] => [b, querysz, 1]
			indices = (indices[...,-1]).contiguous().view(batchsz, querysz, 1)
			# get from [b, querysz, setsz] with index: positive indices: [b, querysz, 1]
			# => [b, querysz, 1] => [b, querysz]
			dist_ap = torch.gather(dist, dim=2, index=indices).view(batchsz, querysz)
			# [b, querysz] - [b, querysz] + margin
			# we use relu function to replace max(x, 0) operation
			# => [b, querysz] => scalar
			loss = F.relu(dist_ap - min_neg + margin).sum() / batchsz
			if np.random.randint(100)<3:
				print('y:', support_y[0].cpu().data.numpy(), query_y[0].cpu().data.numpy())
				print('ap vs an:', dist_ap[0].cpu().data.numpy(), min_neg[0].cpu().data.numpy())
				print('dist:', dist[0].cpu().data.numpy())
			return loss

		else:
			# select the closet element
			# [b, querysz, setsz] => [b, querysz]
			_, indices = dist.min(dim=2)
			# support_y: [b, querysz]
			# indices: [b, querysz]
			# pred: [b, querysz], global true label
			pred = torch.gather(support_y, dim=1, index=indices)

			correct = torch.eq(pred, query_y).sum()
			return pred, correct

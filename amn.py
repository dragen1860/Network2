import torch
from torch import nn
from torch import optim
from torch.autograd import Variable


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
		self.downsample = nn.Sequential(nn.AvgPool2d(7, 5, padding=0))

	def forward(self, x):
		x =  self.net(x)
		# print(x.size())
		return self.downsample(x)



class AMN(nn.Module):
	"""
	Add-Minus-Network: Add all objects information of one image and find out the difference of distinct images.
	"""

	def __init__(self, nway, kshot, kquery, imgsz):
		"""

		:param nway: n way
		:param kshot: k shot
		:param kquery: k query
		:param c: channel depth of feature map
		:param d: objects number of feature map
		"""
		super(AMN, self).__init__()

		self.nway = nway
		self.kshot = kshot
		self.kquery = kquery
		self.v = 256

		self.repnet = Naive()
		repnet_sz = self.repnet(Variable(torch.rand(2, 3, imgsz, imgsz))).size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		assert repnet_sz[2] == repnet_sz[3]
		print('Repnet sz:', repnet_sz, 'V size:', self.v)

		# 1. convert from raw object information to fixed dim vector
		self.g = nn.Sequential(
			nn.Linear(self.c + 2, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, self.v),
			nn.ReLU(inplace=True)
		)
		# 2. sum over all objects information from whole image
		# 3. subtract the vector of two images
		# 3. find the difference between two images via their vector
		self.f = nn.Sequential(
			nn.Linear(self.v, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 1)
		)

		self.criteon = nn.CrossEntropyLoss()


	def forward(self, support_x, support_y, query_x, query_y, train = True):
		"""

		:param support_x:   [b, setsz, c_, h, w]
		:param support_y:   [b, setsz]
		:param query_x:     [b, querysz, c_, h, w]
		:param query_y:     [b, querysz]
		:param train:
		:return:
		"""
		batchsz, setsz, c_, h, w = support_x.size()
		querysz = query_x.size(1)
		c, d, v = self.c, self.d, self.v

		# 1. retain feature map
		# => [b, setsz, c, d, d]
		support_f = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
		# => [b, querysz, c, d, d]
		query_f = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

		# 2. convert objects info
		# [b, setsz, c, d:1, d:2] => [b, setsz, d:1, d:2, c] => [b*setsz*d*d, c]
		support_f = support_f.permute(0, 1, 3, 4, 2).view(batchsz * setsz * d * d, c)
		# [b, querysz, c, d:1, d:2] => [b, querysz, d:1, d:2, c] => [b*querysz*d*d, c]
		query_f = query_f.permute(0, 1, 3, 4, 2).view(batchsz * querysz * d * d, c)
		# [b*setsz*d*d, c] => [b*setsz*d*d, v] => [b, setsz, d*d, v]
		support_f = self.g(support_f).view(batchsz, setsz, d * d, v)
		# [b*querysz*d*d, c] => [b*querysz*d*d, v] => [b, querysz, d*d, v]
		query_f = self.g(query_f).view(batchsz, querysz, d * d, v)

		# 3. sum over all objects info from whole image
		# [b, setsz, d*d, v] => [b, setsz, v]
		support_f = support_f.sum(2)
		# [b, querysz, d*d, v] => [b, querysz, v]
		query_f = query_f.sum(2)

		# 4. difference of distinct image pairs
		# [b, setsz, v] => [b, 1, setsz, v] => [b, querysz, setsz, v]
		support_f = support_f.unsqueeze(1).expand(batchsz, querysz, setsz, v)
		# [b, querysz, v] => [b, querysz, 1, v] => [b, querysz, setsz, v]
		query_f = query_f.unsqueeze(2).expand(batchsz, querysz, setsz, v)
		# => [b, querysz, setsz, v]
		diff = support_f - query_f

		# 5. classify
		# TODO:
		# [b, querysz, setsz, v] => [b, querysz, setsz]
		score = self.f(diff.view(batchsz * querysz * setsz, v)).view(batchsz, querysz, setsz)

		# build pair label
		# [b, setsz] => [b, 1, setsz] => [b, querysz, setsz]
		support_yf = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
		# [b, querysz] => [b, querysz, 1] => [b, querysz, setsz]
		query_yf = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
		# eq: [b, querysz, setsz] => [b, querysz, setsz] and convert byte tensor to float tensor
		label = torch.eq(support_yf, query_yf).float()

		# compute loss
		loss = self.criteon(score.view(batchsz * querysz * setsz, self.nway), label.view(batchsz * querysz * setsz))

		# [b, querysz, setsz] => [b, querysz]
		_, indices = score.max(dim = 2)
		# support_y: [b, setsz], setsz = n-way
		# indices: [b, querysz]
		# pred: [b, querysz], global true label
		pred = torch.gather(support_y, dim=1, index=indices)

		correct = torch.eq(pred, query_y).sum()
		return loss, pred, correct
























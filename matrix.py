import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from repnet import repnet_deep, Bottleneck


class Matrix(nn.Module):
	"""

	"""
	def __init__(self, n_way, k_shot):
		super(Matrix, self).__init__()

		self.n_way = n_way
		self.k_shot = k_shot

		# (256, 4, 4)
		self.repnet = nn.Sequential(repnet_deep(False), # (1024, 14, 14)
		                            nn.Conv2d(1024, 256, kernel_size=5, stride=3),
		                            nn.BatchNorm2d(256),
		                            nn.ReLU(inplace=True))
		# we need to know the feature dim, so here is a forwarding.
		repnet_sz = self.repnet(Variable(torch.rand(2, 3, 224, 224))).size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		# this is the input channels of layer4&layer5
		self.inplanes = 2 * self.c
		assert repnet_sz[2] == repnet_sz[3]
		print('repnet sz:', repnet_sz)

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
		                       nn.Linear(256, 29),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(29, 1),
		                       nn.Sigmoid())

		coord = np.array([(i / self.d , j / self.d) for i in range(self.d) for j in range(self.d)])
		self.coord = torch.from_numpy(coord).float().view(self.d, self.d, 2).transpose(0, 2).transpose(1,2).contiguous()
		self.coord = self.coord.unsqueeze(0).unsqueeze(0)
		print('self.coord:', self.coord.size(),self.coord) # [batchsz:1, setsz:1, 2, self.d, self.d]


	def predict(self, support_x, support_y, query_x, query_y):
		"""
		predicting is different from forward
		:param support_x: [b, setsz, c_, h, w]
		:param support_y: [b, setsz]
		:param query_x:   [b, querysz, c_, h, w]
		:param query_y:   [b, querysz]
		:return:
		"""
		batchsz, setsz, c_, h, w = support_x.size()
		querysz = query_x.size(1)
		# [b, setsz, c_, h, w] => [b, 1, setsz, c_, h, w] => [b, querysz, setsz, c_, h, w] => [b*querysz, setsz, c_, h, w]
		support_x = support_x.unsqueeze(1).expand(batchsz, querysz, setsz, c_, h, w).contiguous().view(batchsz * querysz, setsz, c_, h, w)
		# [b, querysz, c_, h, w] => [b, querysz, 1, c_, h, w] => [b*querysz, 1, c_, h, w]
		query_x = query_x.unsqueeze(2).view(batchsz * querysz, 1, c_, h, w)
		# cat [b*querysz, setsz, c_, h, w] with [b*querysz, 1, c_, h, w] => [b*querysz, setsz+1, c_, h, w]
		input = torch.cat([support_x, query_x], dim = 1)

		# get relation matrix
		# input: [b*querysz, setsz+1, c_, h, w]
		# score: [b*querysz, setsz+1, setsz+1] => [b, querysz, setsz+1, setsz+1]
		# this is different forward, whose size: [b, setsz+querysz, setsz+querysz]
		score = self.rn(input).view(batchsz, querysz, setsz + 1, setsz + 1)

		# now try to find the maximum similar node from score matrix
		# [b, querysz, setsz+1, setsz+1]
		score_np = score.cpu().data.numpy()
		pred = []
		support_y_np = support_y.cpu().data.numpy()
		for i, batch in enumerate(score_np): # batch [querysz, setsz+1, setsz+1]
			for j, query in enumerate(batch): # query [setsz+1, setsz+1] 
				row = query[-1, :-1] # [setsz], the last row, all columns from 0 to -1, exclusive
				col = query[:-1, -1] # [setsz], the last column, all rows from 0 to -1, exclusive
				row_sim = col_sim = [] # [n_way]
				for way in range(self.n_way):
					row_sim.append( np.sum(row[way * self.k_shot : (way + 1) * self.k_shot]) )
					col_sim.append( np.sum(col[way * self.k_shot : (way + 1) * self.k_shot]) )
				merge = np.array(row) + np.array(col)  # element-wise add for np.array, not extend for list,
				idx = merge.argmax()
				pred.append(support_y_np[i, idx * self.k_shot])
		# pred: [b, querysz]
		pred = Variable(torch.from_numpy(np.array(pred).reshape((batchsz, querysz)))).cuda()

		# [1]
		correct = torch.eq(pred, query_y).sum()

		return pred, correct




	def rn(self, input):
		"""
		Given the input relation imgs, output a relation matrix.
		This function will be shared by predicting and forwarding fuction since both the two functions requires relation
		matrix to predict and backprop.
		:param input: [b, setsz, c_, h, w]
		:return:
		"""
		batchsz, setsz, c_, h, w = input.size()
		c, d = self.c, self.d  # c will be set when the function runs

		## Combination bewteen two images, between objects in two images
		# get feature from [b, setsz, c_, h, w] => [b*setsz, c, d, d] => [b, setsz, c, d, d]
		x = self.repnet(input.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
		# [b, setsz, c, d, d] => [b, setsz, c+2, d, d]
		x = torch.cat([x, Variable(self.coord.expand(batchsz, setsz, 2, d, d)).cuda()], dim=2)
		# update c, DO not update self.c
		c += 2
		# [b, setsz, c, d, d] => [b, setsz, c, d*d]
		x = x.view(batchsz, setsz, c, d * d)

		# [b, setsz, c, d*d] => [b, 1, setsz, c, d*d] => [b, 1, setsz, c, 1, d*d] => [b, setsz, setsz, c, d*d, d*d]
		x_i = x.unsqueeze(1).unsqueeze(4).expand(batchsz, setsz, setsz, c, d*d, d*d)
		# [b, setsz, c, d*d] => [b, setsz, 1, c, d*d, 1] => [b, setsz, setsz, c, d*d, d*d]
		x_j = x.unsqueeze(2).unsqueeze(5).expand(batchsz, setsz, setsz, c, d*d, d*d)
		# [b, setsz, setsz, 2c, d*d, d*d]
		x_rn = torch.cat([x_i, x_j], dim=3)

		# [b, setsz, setsz, 2c, d*d:0, d*d:1] => [b, setsz, setsz, d*d:0, d*d:1, 2c] => [b*setsz*setsz*d^4, 2c]
		x_rn = x_rn.transpose(3, 5).transpose(3, 4).contiguous().view(batchsz * setsz * setsz * d*d * d*d, c * 2)
		# push to G network
		# [b*setsz*setsz*d^4, 2c] => [b*setsz*setsz*d^4, new_dim]
		x_f = self.g(x_rn)
		# sum over coordinate axis, erase spatial dims => [batchsz * setsz * setsz, -1]
		x_f = x_f.view(batchsz * setsz * setsz, d * d * d * d, -1).sum(1)  # the last dim can be derived by layer setting

		# push to F network
		# [batchsz * setsz * setsz, -1] => [batchsz * setsz * setsz, 1] => [batch, setsz, setsz]
		score = self.f(x_f).view(batchsz, setsz, setsz)

		return score

	def forward(self, support_x, support_y, query_x, query_y, train=True):
		"""
		To satisfy the multi-gpu trainined, we merge predict and train into one forward function.
		:param support_x: [b, setsz, c_, h, w]
		:param support_y: [b, setsz]
		:param query_x:   [b, querysz, c_, h, w]
		:param query_y:   [b, querysz]
		:return:
		"""
		if not train:
			return self.predict(support_x, support_y, query_x, query_y)

		# [b, setsz, c_, h, w] + [b, querysz, c_, h, w] => [batchsz, setsz+querysz, c_, h, w]
		input = torch.cat([support_x, query_x], dim = 1)
		#
		batchsz, setsz, c_, h, w = support_x.size()
		querysz = query_x.size(1)

		# get similarity matrix score
		# [b, setsz+querysz, setsz+querysz]
		score = self.rn(input)

		# build its label
		# cat [b, setsz/querysz] => [b, setsz+querysz]
		input_y = torch.cat([support_y, query_y], dim = 1)
		# [b, setsz+querysz] => [b, 1, setsz+querysz] => [b, setsz+querysz, setsz+querysz]
		input_y_i = input_y.unsqueeze(1).expand(batchsz, querysz + querysz, setsz+querysz)
		# [b, setsz+querysz] => [b, setsz+querysz, 1] => [b, setsz+querysz, setsz+querysz]
		input_y_j = input_y.unsqueeze(2).expand(batchsz, setsz+querysz, setsz+querysz)
		# eq: [b, setsz+querysz, setsz+querysz] => [b, setsz+querysz, setsz+querysz] and convert byte tensor to float tensor
		label = torch.eq(input_y_i, input_y_j).float()

		# score: [b, setsz+querysz, setsz+querysz]
		# label: [b, setsz+querysz, setsz+querysz]
		loss = torch.pow(label - score, 2).sum() / batchsz
		return loss

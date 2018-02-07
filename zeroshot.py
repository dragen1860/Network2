import torch, os
import numpy as np
from torch.autograd import Variable
from torch import  nn
# import pretrainedmodels
from torchvision.models import resnet18

class Zeroshot(nn.Module):

	def __init__(self, n_way, imgsz = 299):
		super(Zeroshot, self).__init__()

		repnet = resnet18(pretrained=True)
		modules = list(repnet.children())[:-2]
		self.repnet = nn.Sequential(*modules)

		# we need to know the feature dim, so here is a forwarding.
		# => [64, 10, 10]
		repnet_sz = self.repnet(Variable(torch.rand(2, 3, imgsz, imgsz)))
		repnet_sz = repnet_sz.size()
		self.c = repnet_sz[1]
		self.d = repnet_sz[2]
		assert repnet_sz[2] == repnet_sz[3]
		print('Repnet sz:', repnet_sz)

		self.att_dim = 514
		self.attnet = nn.Sequential(nn.Linear(312, 512),
		                            nn.ReLU(inplace=True),
		                            nn.Linear(512, self.att_dim),
		                            nn.ReLU(inplace=True))


		# the input is self.c with two coordination information, and then combine each
		# 2c+4 => 256
		self.g = nn.Sequential(nn.Linear( (self.c + 2) + self.att_dim, 256),
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
		                       nn.BatchNorm1d(29),
		                       nn.ReLU(inplace=True),
		                       nn.Linear(29, 1),
		                       nn.Sigmoid())



		coord = np.array([(i / self.d, j / self.d) for i in range(self.d) for j in range(self.d)])
		self.coord = torch.from_numpy(coord).float().view(self.d, self.d, 2).transpose(0, 2).transpose(1,2).contiguous()
		self.coord = self.coord.unsqueeze(0).unsqueeze(0)
		# print('self.coord:', self.coord) # [batchsz:1, n_way:1, 2, self.d, self.d]



	def forward(self, x, x_label, att, att_label, train = True):
		"""

		:param x:           [batchsz, setsz, c_, h, w]
		:param x_label:     [batchsz, setsz]
		:param att:         [batchsz, n_way, 312]
		:param att_label:   [batchsz, n_way]
		:param train: for train or pred
		:return:
		"""
		batchsz, setsz, c_, h, w = x.size()
		n_way = att.size(1)
		c, d = self.c, self.d

		# [b, setsz, c_, h, w] => [b, setsz, c, d, d]
		x_f = self.repnet(x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)

		## now make the combination between image and attribute
		# include the coordinate information in each feature dim firstly
		# [b, setsz, c+2, d, d]
		x_f = torch.cat([x_f, Variable(self.coord.expand(batchsz, setsz, 2, d, d)).cuda()], dim = 2)
		c += 2 # c is a copy of self.c, we need not reset c since it will be reseted in the beginning of forward

		# make combination now
		# [b, setsz, c, d, d] => [b, setsz, c, d*d] => [b, setsz, 1, c, d*d] => [b, setsz, n_way, c, d*d]
		x_f = x_f.view(batchsz, setsz, c, d*d).unsqueeze(2).expand(batchsz, setsz, n_way, c, d*d)
		# [b, n_way, 312] => [b, 1, n_way, 312] => [b, 1, n_way, 312, 1] => [b, setsz, n_way, 312, d*d]
		att_f = self.attnet(att.view(batchsz * n_way, 312)).view(batchsz, n_way, self.att_dim)
		att_f = att_f.unsqueeze(1).unsqueeze(4).expand(batchsz, setsz, n_way, self.att_dim, d*d)
		# [b, setsz, n_way, c + self.att_dim, d*d]
		comb = torch.cat([x_f, att_f], dim=3)
		c += self.att_dim # udpate c

		# [b, setsz, n_way, c, d*d] => [b, setsz, n_way, d*d, c] => [b*setsz*n_way*d*d, c]
		comb = comb.transpose(3,4).contiguous().view(batchsz * setsz * n_way * d*d, c)
		# [b * setsz * n_way * d*d, c] => [b * setsz * n_way * d*d, 256] => [b, setsz, n_way, d*d, -1]
		x_f = self.g(comb).view(batchsz, setsz, n_way, d*d, -1)
		# sum over spatial rn
		# [b, setsz, n_way, d*d, -1] => [b, setsz, n_way, -1]
		x_f = x_f.sum(dim = 3)
		# push to F network
		# [b, setsz, n_way, -1] => [b*setsz*n_way, -1] => [b, setsz, n_way, 1] => [b, setsz, n_way]
		score = self.f(x_f.view(batchsz * setsz * n_way, -1)).view(batchsz, setsz, n_way)

		# build its label
		# [b, setsz] => [b, setsz, 1] => [b, setsz, n_way]
		x_labelf = x_label.unsqueeze(2).expand(batchsz, setsz, n_way)
		# [b, n_way] => [b, 1, n_way] => [b, setsz, n_way]
		att_labelf = att_label.unsqueeze(1).expand(batchsz, setsz, n_way)
		# eq: [b, setsz, n_way] => [b, setsz, n_way] and convert byte tensor to float tensor
		label = torch.eq(x_labelf, att_labelf).float()
		# print(label[0,0])
		# print(score[0,0])



		# score: [b, setsz, n_way]
		# label: [b, setsz, n_way]
		if train:
			loss = torch.pow(label - score, 2).sum()
			return loss

		else:
			# [b, setsz, n_way] => [b, setsz]
			_, indices = score.max(dim = 2)
			# att_label: [b, n_way]
			# indices: [b, setsz]
			# pred: [b, setsz], global true label
			pred = torch.gather(att_label, dim=1, index=indices)
			# print('scor:', score.cpu().data[0].numpy())
			# print('attl:', att_label.cpu().data[0].numpy())
			# print('pred:', pred.cpu().data[0].numpy())
			# print('x  l:', x_label.cpu().data[0].numpy())

			correct = torch.eq(pred, x_label).sum()
			return pred, correct





def test():
	net = Zeroshot(50)
	# whole parameters number
	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total params:', params)



if __name__ == '__main__':
	test()
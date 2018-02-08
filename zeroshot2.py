import torch, os
import numpy as np
from torch.autograd import Variable
from torch import  nn

from cub import Cub
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, resnet18

# save best acc info, to save the best model to ckpt.
best_accuracy = 0
def evaluation(net, n_way, k_query, mdl_file, repnet, imgsz, batchsz):
	"""
	obey the expriment setting of MAML and Learning2Compare, we randomly sample 600 episodes and 15 query images per query
	set.
	:param net:
	:param batchsz:
	:return:
	"""
	# we need to test 11788 - 8855 = 2933 images.
	db = Cub('../CUB_200_2011_ZL/', n_way, k_query, train=False, episode_num= 1000//n_way//k_query, imgsz=imgsz)
	db_loader = DataLoader(db, 1, shuffle=True, num_workers=1, pin_memory=True)

	accs = []
	for batch in db_loader:
		x = Variable(batch[0]).cuda()
		x_label = Variable(batch[1]).cuda()
		att = Variable(batch[2]).cuda()
		att_label = Variable(batch[3]).cuda()

		# prepare for following procedure.
		real_batchsz = x.size(0)
		setsz = x.size(1)

		# [b, setsz, c, h, w] => [b*setsz, c, h, w]
		x = x.view(real_batchsz * setsz, 3, imgsz, imgsz)
		# [small batch, c, h, w]
		x_chunks = torch.chunk(x, batchsz * n_way, dim=0)
		features = []
		for img in x_chunks:
			# [small batch, 512, 1, 1] => [small batch, 512]
			feature = repnet(img).view(img.size(0), 512)
			features.append(feature)
		# [b*setsz, 512] => [real batch, setsz, 512]
		x = torch.cat(features, dim=0).view(real_batchsz, setsz, 512)
		# detach gradient !!!
		x = x.detach()

		pred, correct = net(x, x_label, att, att_label, False)
		correct = correct.sum().data[0] # multi-gpu

		# preds = torch.cat(preds, dim= 1)
		acc = correct / ( x_label.size(0) * x_label.size(1) )
		accs.append(acc)

		# if np.random.randint(10)<1:
		# 	print(pred[0].cpu().data.numpy(), att_label[0].cpu().data.numpy())
	print(accs)

	# compute the distribution of 600/episodesz episodes acc.
	global best_accuracy
	accuracy = np.array(accs).mean()
	print('<<<<<<<<< %d way accuracy:'%n_way, accuracy, 'best accuracy:', best_accuracy, '>>>>>>>>')

	if accuracy > best_accuracy:
		best_accuracy = accuracy
		torch.save(net.state_dict(), mdl_file)
		print('Saved to checkpoint:', mdl_file)

	return accuracy

class Zeroshot(nn.Module):


	def __init__(self):
		super(Zeroshot, self).__init__()

		self.net = nn.Sequential(nn.Linear(312, 512),
		                         nn.ReLU(inplace=True),
		                         nn.Linear(512, 512),
		                         nn.ReLU(inplace=True))


	def forward(self, x, x_label, att, att_label, train=True):
		"""

		:param x:           [b, setsz, c]
		:param x_label:     [b, setsz]
		:param att:         [b, n_way, 312]
		:param att_label:   [b, n_way]
		:return:
		"""
		batchsz, setsz, c= x.size()
		n_way = att.size(1)

		# [b, setsz, c] => [b*setsz, c]
		x = x.view(batchsz * setsz, c)
		# [b, n_way, 312] => [b*nway, 312]
		att = att.view(batchsz * n_way, 312)
		# [b*nway, 312] => [b*nway, 512]
		att = self.net(att)

		if train:
			# x [b*setsz, 512]
			# att: [b*nway, 512]
			loss = torch.pow(x - att, 2).sum() / (batchsz * setsz)

			return loss

		else:
			x = x.view(batchsz, setsz, c)
			att = att.view(batchsz, n_way, 512)

			x = x.unsqueeze(2).expand(batchsz, setsz, n_way, c)
			att = att.unsqueeze(1).expand(batchsz, setsz, n_way, c)
			# [b, setsz, n, c] => [b, setsz, n] => [b, setsz]
			_, indices = torch.pow(x - att, 2).sum(3).min(2)
			# [b, setsz]
			pred = torch.gather(att_label, 1, index=indices)
			correct = torch.eq(pred, x_label).sum()
			return pred, correct



def test():

	batchsz = 1
	n_way = 50
	k_query = 1
	lr = 1e-3
	imgsz = 299
	mdl_file = 'ckpt/cub2.mdl'


	net = Zeroshot().cuda()
	if os.path.exists(mdl_file):
		print('load from checkpoint ...', mdl_file)
		net.load_state_dict(torch.load(mdl_file))
	else:
		print('training from scratch.')


	optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)

	# 1000, last layer, 512x1x1 last last layer.
	repnet = inception_v3(pretrained=True)
	modules = list(repnet.children())[:-2]
	repnet = nn.Sequential(*modules).cuda()

	for epoch in range(1000):
		db = Cub('../CUB_200_2011_ZL/', n_way, k_query, train=True, imgsz=imgsz)
		db_loader = DataLoader(db, batchsz, shuffle=True, num_workers=2, pin_memory=True)
		total_train_loss = 0

		for step, batch in enumerate(db_loader):

			if step % 300 == 0:
				# evaluation(net, n_way, k_query, mdl_file, repnet, imgsz, batchsz):
				# evaluation(net, n_way, k_query, mdl_file, repnet, imgsz, batchsz)
				pass

			# 2. train
			x = Variable(batch[0]).cuda()
			x_label = Variable(batch[1]).cuda()
			att = Variable(batch[2]).cuda()
			att_label = Variable(batch[3]).cuda()

			# prepare for following procedure.
			real_batchsz = x.size(0)
			setsz = x.size(1)

			# [b, setsz, c, h, w] => [b*setsz, c, h, w]
			x = x.view(real_batchsz * setsz, 3, imgsz, imgsz)
			# [small batch, c, h, w]
			x_chunks = torch.chunk(x, 2, dim = 0)
			features = []
			for img in x_chunks:
				# [small batch, 512, 1, 1] => [small batch, 512]
				print(img.size())
				feature = repnet(img)
				print(feature.size())
				feature = feature.view(img.size(0), 512)
				features.append(feature)
			# [b*setsz, 512] => [real batch, setsz, 512]
			x = torch.cat(features, dim= 0 ).view(real_batchsz, setsz, 512)
			# detach gradient !!!
			x = x.detach()

			net.train()
			loss = net(x, x_label, att, att_label)
			total_train_loss += loss.data[0]

			optimizer.zero_grad()
			loss.backward()
			# if np.random.randint(1000)<2:
			# 	for p in net.parameters():
			# 		print(p.grad.norm(2).data[0])
			nn.utils.clip_grad_norm(net.parameters(), 1)
			optimizer.step()

			# 3. print
			if step % 20 == 0 and step != 0:
				print('%d-way %d batch> epoch:%d step:%d, loss:%f' % (
				n_way,  batchsz, epoch, step, loss.data[0]) )
				total_train_loss = 0



if __name__ == '__main__':
	test()
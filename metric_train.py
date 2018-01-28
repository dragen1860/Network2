import torch, os
import numpy as np
from torch import optim
from torch.autograd import Variable
from MiniImagenet import MiniImagenet
from metric import Metric
from utils import make_imgs

if __name__ == '__main__':
	from MiniImagenet import MiniImagenet
	from torch.utils.data import DataLoader
	from torchvision.utils import make_grid
	from torch.optim import lr_scheduler
	from tensorboardX import SummaryWriter
	from datetime import datetime
	import random
	import argparse


	argparser = argparse.ArgumentParser()
	argparser.add_argument('-n', help='n way', default=5)
	argparser.add_argument('-k', help='k shot', default=1)
	argparser.add_argument('-b', help='batch size', default=1)
	argparser.add_argument('-l', help='learning rate', default=1e-3)
	args = argparser.parse_args()
	n_way = int(args.n)
	k_shot = int(args.k)
	batchsz = int(args.b)
	lr = float(args.l)
	k_query = 1
	imgsz = 224

	torch.manual_seed(66)
	np.random.seed(66)
	random.seed(66)

	net = Metric(n_way, k_shot, imgsz).cuda()
	print(net)
	mdl_file = 'ckpt/metric%d%d.mdl'%(n_way, k_shot)
	print('mini-imagnet: %d-way %d-shot lr:%f' % (n_way, k_shot, lr))

	if os.path.exists(mdl_file):
		print('load checkpoint ...', mdl_file)
		net.load_state_dict(torch.load(mdl_file), strict=False)
	else:
		print('training from scratch.')

	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total params:', params)

	optimizer = optim.Adam(net.parameters(), lr=lr)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

	best_accuracy = 0
	for epoch in range(1000):
		mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=10000, resize=imgsz)
		db = DataLoader(mini, batchsz, shuffle=True, num_workers=8, pin_memory=True)
		mini_val = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                        batchsz=200, resize=imgsz)
		db_val = DataLoader(mini_val, batchsz, shuffle=True, num_workers=4, pin_memory=True)
		total_train_loss = 0

		for step, batch in enumerate(db):

			# 1. test
			total_val_loss = 0
			if step % 300 == 0:
				total_correct = 0
				total_num = 0

				for j, batch_test in enumerate(db_val):
					support_x = Variable(batch_test[0]).cuda()
					support_y = Variable(batch_test[1]).cuda()
					query_x = Variable(batch_test[2]).cuda()
					query_y = Variable(batch_test[3]).cuda()

					net.eval()
					pred, correct = net(support_x, support_y, query_x, query_y, False)
					total_correct += correct.data[0]
					total_num += query_y.size(0) * query_y.size(1)


				accuracy = total_correct / total_num
				if accuracy > best_accuracy :
					best_accuracy = accuracy
					torch.save(net.state_dict(), mdl_file)
					torch.save(net, mdl_file[:-4]+'.whl')
					print('Saved to checkpoint and whole mdl! ', mdl_file, mdl_file[:-4]+'.whl')

				print('<<<<>>>>accuracy:', accuracy, 'best accuracy:', best_accuracy)

				# scheduler.step(accuracy)

			# 2. train
			support_x = Variable(batch[0]).cuda()
			support_y = Variable(batch[1]).cuda()
			query_x = Variable(batch[2]).cuda()
			query_y = Variable(batch[3]).cuda()

			net.train()
			loss = net(support_x, support_y, query_x, query_y)
			total_train_loss += loss.data[0]

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# 3. print
			if step % 20 == 0 and step != 0:
				print('%d-way %d-shot %d batch> epoch:%d step:%d, loss:%f' % (
				n_way, k_shot, batchsz, epoch, step, total_train_loss) )
				total_train_loss = 0

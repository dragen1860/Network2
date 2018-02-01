import torch, os
import numpy as np
from torch import optim
from torch.autograd import Variable
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys
import argparse
from torch import  nn
from compdeep import CompDeep



def mean_confidence_interval(accs, confidence = 0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf( (1 + confidence) / 2, n - 1)
    return m, h


# save best acc info, to save the best model to ckpt.
best_accuracy = 0
def evaluation(net, batchsz, n_way, k_shot, imgsz, episodesz, threhold, mdl_file):
	"""
	obey the expriment setting of MAML and Learning2Compare, we randomly sample 600 episodes and 15 query images per query
	set.
	:param net:
	:param batchsz:
	:return:
	"""
	k_query = 15
	mini_val = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query,
	                        batchsz=600, resize=imgsz)
	db_val = DataLoader(mini_val, batchsz, shuffle=True, num_workers=2, pin_memory=True)

	accs = []
	episode_num = 0 # record tested num of episodes

	for batch_test in db_val:
		# [60, setsz, c_, h, w]
		# setsz = (5 + 15) * 5
		support_x = Variable(batch_test[0]).cuda()
		support_y = Variable(batch_test[1]).cuda()
		query_x = Variable(batch_test[2]).cuda()
		query_y = Variable(batch_test[3]).cuda()

		# we will split query set into 15 splits.
		# query_x : [batch, 15*way, c_, h, w]
		# query_x_b : tuple, 15 * [b, way, c_, h, w]
		query_x_b = torch.chunk(query_x, k_query, dim= 1)
		# query_y : [batch, 15*way]
		# query_y_b: 15* [b, way]
		query_y_b = torch.chunk(query_y, k_query, dim= 1)
		preds = []
		net.eval()
		# we don't need the total acc on 600 episodes, but we need the acc per sets of 15*nway setsz.
		total_correct = 0
		total_num = 0
		for query_x_mini, query_y_mini in zip(query_x_b, query_y_b):
			# print('query_x_mini', query_x_mini.size(), 'query_y_mini', query_y_mini.size())
			pred, correct = net(support_x, support_y, query_x_mini.contiguous(), query_y_mini, False)
			correct = correct.sum() # multi-gpu
			# pred: [b, nway]
			preds.append(pred)
			total_correct += correct.data[0]
			total_num += query_y_mini.size(0) * query_y_mini.size(1)
		# # 15 * [b, nway] => [b, 15*nway]
		# preds = torch.cat(preds, dim= 1)
		acc = total_correct / total_num
		print('%.5f,'%acc, end=' ')
		sys.stdout.flush()
		accs.append(acc)

		# update tested episode number
		episode_num += query_y.size(0)
		if episode_num > episodesz:
			# test current tested episodes acc.
			acc = np.array(accs).mean()
			if acc >= threhold:
				# if current acc is very high, we conduct all 600 episodes testing.
				continue
			else:
				# current acc is low, just conduct `episodesz` num of episodes.
				break


	# compute the distribution of 600/episodesz episodes acc.
	global best_accuracy
	accs = np.array(accs)
	accuracy, sem = mean_confidence_interval(accs)
	print('\naccuracy:', accuracy, 'sem:', sem)
	print('<<<<<<<<< accuracy:', accuracy, 'best accuracy:', best_accuracy, '>>>>>>>>')

	if accuracy > best_accuracy:
		best_accuracy = accuracy
		torch.save(net.state_dict(), mdl_file)
		print('Saved to checkpoint:', mdl_file)

	return accuracy, sem


def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-n', help='n way')
	argparser.add_argument('-k', help='k shot')
	argparser.add_argument('-b', help='batch size')
	argparser.add_argument('-l', help='learning rate', default=1e-3)
	args = argparser.parse_args()
	n_way = int(args.n)
	k_shot = int(args.k)
	batchsz = int(args.b)
	lr = float(args.l)

	k_query = 1
	imgsz = 224
	threhold = 0.7 if k_shot==5 else 0.59 # threshold for when to test full version of episode
	mdl_file = 'ckpt/compdeep%d%d.mdl'%(n_way, k_shot)
	print('mini-imagnet: %d-way %d-shot lr:%f, threshold:%f' % (n_way, k_shot, lr, threhold))

	# torch.manual_seed(66)
	# np.random.seed(66)
	# random.seed(66)


	net = nn.DataParallel(CompDeep(n_way, k_shot, imgsz), device_ids=[0]).cuda()
	print(net)

	if os.path.exists(mdl_file):
		print('load from checkpoint ...', mdl_file)
		net.load_state_dict(torch.load(mdl_file))
	else:
		print('training from scratch.')

	# whole parameters number
	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Total params:', params)

	# build optimizer and lr scheduler
	optimizer = optim.Adam(net.parameters(), lr=lr)
	# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=20, verbose=True)

	for epoch in range(1000):
		mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=10000, resize=imgsz)
		db = DataLoader(mini, batchsz, shuffle=True, num_workers=8, pin_memory=True)
		total_train_loss = 0

		for step, batch in enumerate(db):
			# 1. test
			if step % 300 == 0:
				# evaluation(net, batchsz, n_way, k_shot, imgsz, episodesz, threhold, mdl_file):
				accuracy, sem = evaluation(net, batchsz, n_way, k_shot, imgsz, 100, threhold, mdl_file)
				scheduler.step(accuracy)

			# 2. train
			support_x = Variable(batch[0]).cuda()
			support_y = Variable(batch[1]).cuda()
			query_x = Variable(batch[2]).cuda()
			query_y = Variable(batch[3]).cuda()

			net.train()
			loss = net(support_x, support_y, query_x, query_y)
			loss = loss.sum() / support_x.size(0) # multi-gpu, divide by total batchsz
			total_train_loss += loss.data[0]

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# 3. print
			if step % 20 == 0 and step != 0:
				print('%d-way %d-shot %d batch> epoch:%d step:%d, loss:%f' % (
				n_way, k_shot, batchsz, epoch, step, total_train_loss) )
				total_train_loss = 0



if __name__ == '__main__':
	main()

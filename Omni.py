import torch, os
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import torchvision
from Omni_NaiveRN import NaiveRN
from omniglotNShot import OmniglotNShot
from torchvision.utils import make_grid
from utils import make_imgs
from torch.optim import lr_scheduler
import argparse



def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-n', help='n way')
	argparser.add_argument('-k', help='k shot')
	argparser.add_argument('-b', help='batch size')
	args = argparser.parse_args()
	n_way = int(args.n)
	k_shot = int(args.k)
	k_query = 1
	batchsz = int(args.b)
	imgsz = 84

	db = OmniglotNShot('dataset', batchsz=batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query)
	print('OmniglotNShot %d-way %d-shot learning' % (n_way, k_shot))
	net = NaiveRN(n_way, k_shot, imgsz).cuda()
	print(net)
	mdl_file = 'ckpt/omni%d%d.mdl'%(n_way, k_shot)

	if os.path.exists(mdl_file):
		print('load checkpoint ...', mdl_file)
		net.load_state_dict(torch.load(mdl_file))

	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)

	input, input_y, query, query_y = db.get_batch('train')  # (batch, n_way*k_shot, img)
	print('get_batch:', input.shape, query.shape, input_y.shape, query_y.shape)

	optimizer = optim.Adam(net.parameters(), lr=1e-3)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)
	tb = SummaryWriter('runs')

	total_train_loss = 0
	best_accuracy = 0
	for step in range(100000000):

		support_x, support_y, query_x, query_y = db.get_batch('train')
		support_x = Variable(torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		query_x = Variable(torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1)).cuda()
		support_y = Variable(torch.from_numpy(support_y).int()).cuda()
		query_y = Variable(torch.from_numpy(query_y).int()).cuda()

		loss = net(support_x, support_y, query_x,  query_y)
		total_train_loss += loss.data[0]

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		if step % 400 == 0:
			total_correct = 0
			total_num = 0
			total_set_num = 0 # we only test 600 episodes in total
			display_onebatch = False  # display one batch on tensorboard

			while total_set_num < 300:
				support_x, support_y, query_x, query_y = db.get_batch('test')
				support_x = Variable(torch.from_numpy(support_x).float().transpose(2,4).transpose(3, 4).repeat(1,1,3,1,1)).cuda()
				query_x = Variable(torch.from_numpy(query_x).float().transpose(2,4).transpose(3, 4).repeat(1,1,3,1,1)).cuda()
				support_y = Variable(torch.from_numpy(support_y).int()).cuda()
				query_y = Variable(torch.from_numpy(query_y).int()).cuda()

				net.eval()
				pred, correct = net(support_x, support_y, query_x, query_y, False)
				total_correct += correct.data[0]
				total_num += query_y.size(0) * query_y.size(1)


				if not display_onebatch:
					display_onebatch = True  # only display once
					all_img, max_width = make_imgs(n_way, k_shot, k_query, support_x.size(0),
					                               support_x, support_y, query_x, query_y, pred)
					all_img = make_grid(all_img, nrow=max_width)
					tb.add_image('result batch', all_img)

				total_set_num += batchsz

			accuracy = total_correct / total_num
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				torch.save(net.state_dict(), mdl_file)
				print('Saved to checkpoint:', mdl_file)

			tb.add_scalar('accuracy', accuracy)
			print('<<<<>>>>accuracy:', accuracy, 'best accuracy:', best_accuracy)

			scheduler.step(accuracy)

		if step % 20 == 0 and step != 0:
			tb.add_scalar('loss', total_train_loss)
			print('%d-way %d-shot %d batch> step:%d, loss:%f' % (
				n_way, k_shot, batchsz, step, total_train_loss))
			total_train_loss = 0
















if __name__ == '__main__':
	main()

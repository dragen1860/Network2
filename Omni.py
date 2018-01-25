from omniglotNShot import OmniglotNShot
import torch
from torch import optim
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import torchvision
from naivern import NaiveRN


def main():

	n_way = 5
	k_shot = 1
	batch_size = 2
	imgsz = 28
	db = OmniglotNShot('dataset', batch_size=batch_size, samples_per_class=k_shot, classes_per_set=n_way)
	print('%d-way %d-shot learning' % (n_way, k_shot))
	rn = NaiveRN(n_way, k_shot, imgsz).cuda()

	input, input_y, query, query_y = db.get_batch('train')  # (batch, n_way*k_shot, img)
	print('get_batch:', input.shape, query.shape, input_y.shape, query_y.shape)

	optimizer = optim.Adam(rn.parameters(), lr=1e-3)
	tb = SummaryWriter('runs')

	for epoch in range(1000000):
		input, input_y, query, query_y = db.get_batch('train')
		input = Variable(torch.from_numpy(input).float()).cuda()
		query = Variable(torch.from_numpy(query).float()).cuda()
		input_y = Variable(torch.from_numpy(input_y).float()).cuda()  # [batch, setsz, 1]
		query_y = Variable(torch.from_numpy(query_y).float()).cuda()
		loss, _ = rn(input, input_y, query,  query_y)

		if epoch % 20 == 0:
			print(epoch, loss.cpu().data[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


	for epoch in range(1000000):
		input, input_y, query, query_y = db.get_batch('train')
		input = Variable(torch.from_numpy(input).float().transpose(2,4).repeat(1,1,3,1,1)).cuda()
		query = Variable(torch.from_numpy(query).float().transpose(2,4).repeat(1,1,3,1,1)).cuda()
		input_y = Variable(torch.from_numpy(input_y).float()).cuda()  # [batch, setsz, 1]
		query_y = Variable(torch.from_numpy(query_y).float()).cuda()
		loss, _ = rn.forward(input, input_y, query,  query_y)

		if epoch % 20 == 0:
			print(epoch, loss.cpu().data[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if False:
			total_accuracy = 0
			batch_size_test = 100
			for i in range(batch_size_test // batch_size):  # 100  sets
				input, input_y, query, query_y = db.get_batch('val')

				input = Variable(torch.from_numpy(input).float()).cuda()
				query = Variable(torch.from_numpy(query).float()).cuda()
				input_y = Variable(torch.from_numpy(input_y).unsqueeze(2).float()).cuda()
				query_y = Variable(torch.from_numpy(query_y).unsqueeze(2).float()).cuda()
				accuracy, query_pred = rn.predict(input, query, input_y, query_y)

				total_accuracy += accuracy.data[0]

				batchidx = np.random.randint(batch_size)
				input = input[batchidx]  # [setsz, h, w, c]
				input_y = input_y[batchidx]  # [setsz, 1]
				query = query[batchidx]
				query_y = query_y[batchidx]
				query_pred = query_pred[batchidx]

				# make_grid of meta-training, sort them firstly
				input_y_sorted, input_y_sorted_idx = torch.sort(input_y.squeeze(1), dim=0)  # [setsz]
				input_sorted = torch.index_select(input, dim=0, index=input_y_sorted_idx)
				imgs = torchvision.utils.make_grid(
					input_sorted.data.transpose(1, 3).repeat(1, 3, 1, 1))  # accept tensor
				# make_grid of meta-testing, put them in corresponding pos
				query_imgs = torchvision.utils.make_grid(query.data.transpose(1, 3).repeat(2, 3, 1, 1))

				tb.add_image('meta-train', imgs)
				tb.add_image('meta-test', query_imgs)
				tb.add_text('test pred:',
				            str(query_pred.cpu().data.numpy()) + ' == gt:' + str(query_y.cpu().data.numpy()))
				print(str(query_pred.cpu().data.numpy()) + ' == gt:' + str(query_y.cpu().data.numpy()))

			tb.add_scalar('accuracy', total_accuracy / batch_size_test * batch_size)

			print('accuracy:', total_accuracy / batch_size_test * batch_size)


if __name__ == '__main__':
	main()

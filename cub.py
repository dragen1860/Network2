import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
from scipy import io
from torch.utils.data import DataLoader


class Cub(Dataset):
	"""
	images.mat['images'][0,0]: 1-11788
	images.mat['images'][0,1]:  [[array(['001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'],
      dtype='<U61')],
       [array(['001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg'],
      dtype='<U61')], ... ]
	image_class_labels['imageClassLabels']: array([[    1,     1],
											       [    2,     1],
											       [    3,     1],
											       ...,
											       [11786,   200],
											       [11787,   200],
											       [11788,   200]], dtype=int32), shape = [11788, 2]
	class_attribute_labels_continuous['classAttributes'].shape= (200, 312)

	"""
	def __init__(self, root, n_way, k_query, train = True, episode_num = 1000, imgsz = 224):
		"""
		Actually, the image here act as query role. we want to find the closed attribute item for each image.
		:param root:
		:param n_way:
		:param k_query:
		:param train:
		:param episode_num:
		"""
		super(Cub, self).__init__()

		self.root = root
		self.n_way = n_way
		self.k_query = k_query
		self.episode_num = episode_num
		self.train = train
		print('train?', train, '%d-way,'%n_way, '%d-query,'%k_query, '%d-episodes'%episode_num)

		self.img_label = io.loadmat(os.path.join(root, 'image_class_labels.mat'))
		# [1, 1], [2, 1]
		self.img_label = self.img_label['imageClassLabels'][:, 1]
		self.img_label = self.img_label.reshape(11788)
		self.img_label -= 1
		# print('>>img_label:', self.img_label.shape)

		self.img = io.loadmat(os.path.join(root, 'images.mat'))
		# ([1~11788], [img1, img2])
		self.img = self.img['images'][0, 1]
		self.img = np.array(self.img.tolist()).squeeze().reshape(11788)
		# print('>>img:', self.img.shape)



		self.imgdata = io.loadmat(os.path.join(root, 'cnn_feat-imagenet-vgg-verydeep-19.mat'))
		# ([1~11788], [img1, img2])
		self.imgdata = self.imgdata['cnn_feat'].swapaxes(0,1)
		print('cnn_feat-imagenet-vgg-verydeep-19:', self.imgdata.shape)

		# self.imgdata = np.load(open(os.path.join(root, 'features_res34_512.npy'), 'rb'))
		# print('features_res34_512:', self.imgdata.shape)

		img_by_cls = []
		for i in range(200):
			current_img = self.img[np.equal(self.img_label, i)]
			img_by_cls.append(current_img)
		# gathere db by same label: [[label1_img1, label1_img2,...], [label2_img1, label2_img2,...], ...]
		# each class has different num of imgs, here we use a list to save it.
		# can not shuffle!
		self.img_by_cls = img_by_cls[:150] if train else img_by_cls[150:]
		# print('>>img by cls:', len(self.img_by_cls))

		self.att = io.loadmat(os.path.join(root, 'class_attribute_labels_continuous.mat'))
		self.att = self.att['classAttributes'].reshape(200, 312).astype(np.float32)
		# can NOT shuffle
		self.att = self.att[:150] if train else self.att[150:]
		# print('>>att:', self.att.shape)

		# NOTICE:
		# img_cls is order by label and att is order by label as well
		# img_cls: [150/50, img_num]
		# att: [150/50, 312]
		# we don't need label since they are corresponding to each other and we can treat the index as label.


		self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
		                                     transforms.Resize((imgsz, imgsz)),
		                                     transforms.ToTensor(),
		                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		                                     ])

	def __getitem__(self, item):

		# randomly sample n-way classes from train/test set
		# [n_way]
		selected_cls_idx = np.random.choice(range(len(self.img_by_cls)), self.n_way, False)

		# select all imgs for the selected categories, its a list.
		# [n_way, img_num]
		# [[5_img1, 5_img2,...], [29_img1, 29_img2,...],...]
		selected_img_by_cls = [self.img_by_cls[i] for i in selected_cls_idx]
		# only sample one for each category, [[5_img1, 5_img2], [29_img1, 29_img2], ....]
		# [n_way, k_query]
		selected_imgs = [np.random.choice(imgs, self.k_query, False) for imgs in selected_img_by_cls]

		# select attributes for each class
		# [n_way, 312]
		selected_atts = self.att[selected_cls_idx]


		# [n_way, k_query] => [setsz=n_way*k_query]
		selected_imgs = np.array(selected_imgs).reshape(-1)

		# convert relative path to global path to read img by PIL
		# selected_imgs = [os.path.join(self.root,'images', path) for path in selected_imgs]
		x = []
		for img in selected_imgs:
			# find the index in self.img which match current filename
			idx = np.where(self.img == img)[0][0]
			# 1280x8x8
			imgdata = self.imgdata[idx]
			x.append(imgdata)
		# Nx1280x8x8
		x = np.array(x).astype(np.float32)
		x = torch.from_numpy(x)

		att = torch.from_numpy(selected_atts)
		att_label = torch.from_numpy(selected_cls_idx)

		# [n_way] => [n_way, 1] => [n_way, k_query] => [n_way*k_query]
		x_label = att_label.clone().unsqueeze(1).repeat(1, self.k_query).view(-1)


		# shuffle
		shuffle_idx = torch.randperm(self.n_way * self.k_query)
		x = x[shuffle_idx]
		x_label = x_label[shuffle_idx]


		# print('\nselected_imgs', selected_imgs)
		# print('imgs:', x.size())
		# print('attrs:', att.size(), att[:5])
		# print('att label:', att_label.numpy())
		# print('x label:', x_label.numpy())


		return x, x_label, att, att_label



	def __len__(self):
		return self.episode_num





def test():
	db = Cub('../CUB_200_2011_ZL/', 5, 2, train=False)

	db_loader = DataLoader(db, 1, True, num_workers=1, pin_memory=True)

	iter(db_loader).next()


if __name__ == '__main__':
	test()
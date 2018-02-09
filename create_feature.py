import numpy as np
from scipy import io
import os, torch
from torchvision.models import resnet34, resnet50
from torch import nn
from torch.autograd import Variable
from myinception import inception_v3
from PIL import Image
from torchvision import transforms


root = '../CUB_200_2011_ZL'
imgsz = 224

repnet = resnet34(pretrained=True)
modules = list(repnet.children())[:-2] # 512x7x7, 512
repnet = nn.Sequential(*modules)
repnet.cuda()
repnet.eval()

demo = Variable(torch.zeros(1, 3, imgsz, imgsz), volatile = True).cuda()
demo = repnet(demo)
print(demo.size())



imgs = io.loadmat(os.path.join(root, 'images.mat'))
# ([1~11788], [img1, img2])
imgs = imgs['images'][0, 1]
imgs = np.array(imgs.tolist()).squeeze().reshape(11788)

img_label = io.loadmat(os.path.join(root, 'image_class_labels.mat'))
# [1, 1], [2, 1]
img_label = img_label['imageClassLabels'][:, 1]
img_label = img_label.reshape(11788)
img_label -= 1
# print('>>img_label:', self.img_label.shape)

buff = []
transform = transforms.Compose([
	lambda x:Image.open(os.path.join(root, 'images', x)).convert('RGB'),
	transforms.Resize((imgsz, imgsz)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
for i, img in enumerate(imgs):
	print(i, img)

	img = transform(img)
	img = Variable(img.unsqueeze(0)).cuda()
	# 1280x8x8
	img = repnet(img)
	buff.append(img.squeeze().cpu().data.numpy())
buff = np.array(buff).astype(np.float32)
np.save(open(os.path.join(root, 'features_res34_512x7x7.npy'), 'wb'), buff)





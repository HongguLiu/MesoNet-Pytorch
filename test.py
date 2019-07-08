from __future__ import print_function, division

import torch
import torch.nn as nn
import argparse
import time
import os
import cv2
from PIL import Image as pil_image
from tqdm import tqdm
from network.classifier import *
from network.transform import mesonet_data_transforms

def preprocess_image(image, cuda=True):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	preprocess = mesonet_data_transforms['test']
	preprocessed_image = preprocess(pil_image.fromarray(image))

	preprocessed_image = preprocessed_image.unsqueeze(0)
	if cuda:
		preprocessed_image = preprocessed_image.cuda()
	return preprocessed_image

def predict_with_model(image, model, post_function=nn.Softmax(dim=1), cuda=True):
	preprocessed_image = preprocess_image(image, cuda)
	output = model(preprocessed_image)
	output = post_function(output)

	_, prediction = torch.max(output, 1)
	prediction = float(prediction.cpu().numpy())

	return int(prediction), output


def test_images(images_path, model_path, cuda=True):
	if model_path is not None:
		model = torch.load(model_path)
		print('Model found in {}'.format(model_path))
	else:
		print('No model found, please check it!')
	if cuda:
		model = model.cuda()
	fake_count = 0
	real_count = 0
	images_list = os.listdir(images_path)
	for images in images_list:
		image = cv2.imread(os.path.join(images_path, images))
		prediction, output = predict_with_model(image, model, cuda=cuda)
		print(prediction)
		if prediction == 0:
			fake_count += 1
		else:
			real_count += 1
	print("fake frame is:", fake_count)
	print("real frame is:", real_count)

if __name__ == '__main__':
	p = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument('--images_path', '-i', default=".\\deepfake_database\\val\\df", type=str)
	p.add_argument('--model_path', '-mi', default=".\\output\\Meso4_deepfake.pkl", type=str)
	p.add_argument('--cuda', action='store_true')
	args = p.parse_args()
	test_images(**vars(args))
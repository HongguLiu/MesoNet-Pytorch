import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from torchvision import datasets, models, transforms
from network.classifier import *
from network.transform import mesonet_data_transforms
def main():
	args = parse.parse_args()
	test_path = args.test_path
	batch_size = args.batch_size
	model_path = args.model_path
	torch.backends.cudnn.benchmark=True
	test_dataset = torchvision.datasets.ImageFolder(test_path, transform=mesonet_data_transforms['val'])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
	test_dataset_size = len(test_dataset)
	corrects = 0
	acc = 0
	model = Meso4()
	model.load_state_dict(torch.load(model_path))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	model.eval()
	with torch.no_grad():
		for (image, labels) in test_loader:
			image = image.cuda()
			labels = labels.cuda()
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)
			corrects += torch.sum(preds == labels.data).to(torch.float32)
			print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
		acc = corrects / test_dataset_size
		print('Test Acc: {:.4f}'.format(acc))



if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--batch_size', '-bz', type=int, default=32)
	parse.add_argument('--test_path', '-tp', type=str, default='./deepfake_database/test')
	parse.add_argument('--model_path', '-mp', type=str, default='./output/Mesonet/best.pkl')
	main()
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import argparse
import time
import os
import cv2
from PIL import Image as pil_image
from tqdm import tqdm
from network.classifier import *
from network.transform import mesonet_data_transforms

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
	best_model_wts = model.state_dict()
	best_acc = 0.0
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			running_corrects = 0.0

			for inputs, labels in dataloaders[phase]:
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
				optimizer.zero_grad()

				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)

				if phase == 'train':
					loss.backward()
					optimizer.step()

				running_loss += loss.data.item()
				running_corrects += torch.sum(preds == labels.data).to(torch.float32)
			epoch_loss = running_loss / datasets_sizes[phase]
			epoch_acc = running_corrects / datasets_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()
	print('Best val Acc: {:4f}'.format(best_acc))
	model.load_state_dict(best_model_wts)
	return model



if __name__ == '__main__':
	data_dir = '.\\deepfake_database'
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), mesonet_data_transforms[x]) for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
													batch_size=10,
													shuffle=True,
													num_workers=4) for x in ['train', 'val']}
	datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	model = Meso4()
	model = model.cuda()

	criterion = nn.CrossEntropyLoss()

	optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)

	model = train_model(model=model, criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=20)
	torch.save(model, ".\\output\\Meso4_deepfake.pkl")


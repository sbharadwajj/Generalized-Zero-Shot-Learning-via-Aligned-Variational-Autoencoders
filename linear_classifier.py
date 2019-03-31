import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from PIL import Image
import scipy.io
from tqdm.auto import tqdm
from time import sleep
from sklearn.preprocessing import MinMaxScaler
import argparse
from sklearn.metrics import classification_report,confusion_matrix


from dataset import dataloader, classifier_dataloader
from model import encoder_cada, decoder_cada, Classifier
import argparse

parser = argparse.ArgumentParser(description='Feature extraction for vae-gzsl')
parser.add_argument('--dataset', type=str, default='CUB',
					help='Name of the dataset')
parser.add_argument('--batch_size', type=int, default=50,
					help='The batch size')
parser.add_argument('--epochs', type=int, default=100,
					help='Number of epochs')
parser.add_argument('--latent_size', type=int, default=64,
					help='size of the latent vector')
parser.add_argument('--dataset_path', type=str, default='../../../dataset/xlsa17/data/CUB/',
					help='Name of the dataset')
parser.add_argument('--split', type=str, default='train',
					help='Which feature to generate')
parser.add_argument('--model_path', type=str, default='models/checkpoint_100.pth',
					help='math of pretrained model')
parser.add_argument('--features_save', type=str, default='features_extracted/CUB/',
					help='path to save features')
parser.add_argument('--device', type=str, default='cpu',
					help='cuda or cpu')



class Gzsl_vae():
	"""docstring for Gzsl_vae"""
	def __init__(self, ):
		args = parser.parse_args()
		self.device = torch.device(args.device)

		######################## LOAD DATA #############################
		self.trainval_set = dataloader(root=args.dataset_path,split='trainval', device=self.device)
		#train_set = dataloader(root=args.dataset_path,split='train', device=self.device)
		self.test_set = dataloader(root=args.dataset_path,split='test_unseen', device=self.device)
		self.test_set_seen = dataloader(root=args.dataset_path,split='test_seen', device=self.device)
		#val_set = dataloader(root=args.dataset_path,split='val', device=self.device)		

		self.trainloader = data.DataLoader(self.trainval_set, batch_size=args.batch_size, shuffle=True)
		#self.testloader = data.DataLoader(self.test_set, batch_size=args.batch_size, shuffle=False)		

		self.input_dim = self.trainval_set.__getlen__()
		self.atts_dim = self.test_set.__get_attlen__()
		self.num_classes = self.trainval_set.__totalClasses__()
		
		print(20*('-'))
		print("Input_dimension=%d"%self.input_dim)
		print("Attribute_dimension=%d"%self.atts_dim)
		print("z=%d"%args.latent_size)
		print("num_classes=%d"%self.num_classes)
		print(20*('-'))


		####################### INITIALIZE THE MODEL AND OPTIMIZER #####################
		self.model_encoder = encoder_cada(input_dim=self.input_dim,atts_dim=self.atts_dim,z=args.latent_size).to(self.device)
		self.model_decoder = decoder_cada(input_dim=self.input_dim,atts_dim=self.atts_dim,z=args.latent_size).to(self.device)

		learnable_params = list(self.model_encoder.parameters()) + list(self.model_decoder.parameters())
		self.optimizer = optim.Adam(learnable_params, lr=0.00015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

		self.classifier = Classifier(input_dim=args.latent_size,num_class=self.num_classes)
		self.cls_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, betas=(0.5,0.999))

		print(self.model_encoder)
		print(self.model_decoder)
		print(self.classifier)


		########## LOSS ############
		self.l1_loss = nn.L1Loss(reduction='sum')
		self.lossfunction_classifier =  nn.NLLLoss()

		self.gamma = torch.zeros(1, device=self.device).float()
		self.beta = torch.zeros(1, device=self.device).float()
		self.delta = torch.zeros(1, device=self.device).float()



	def train(self, epoch):
		'''
		This function trains the cada_vae model
		'''	
		if epoch > 5 and epoch < 21:
			self.delta += 0.54
		if epoch > 20 and epoch < 75:
			self.gamma += 0.044
		if epoch < 91:
			self.beta += 0.0026

		trainbar = tqdm(self.trainloader)    
		self.model_encoder.train()
		self.model_decoder.train()
		train_loss = 0

		for batch_idx, (x,y,sig) in enumerate(trainbar):		
			z_img, z_sig, mu_x, logvar_x, mu_sig, logvar_sig = self.model_encoder(x, sig)
			recon_x, recon_sig, sigDecoder_x, xDecoder_sig = self.model_decoder(z_img, z_sig) 
			#loss			
			vae_reconstruction_loss = self.l1_loss(recon_x, x) + self.l1_loss(recon_sig, sig)
			cross_reconstruction_loss = self.l1_loss(xDecoder_sig, x) + self.l1_loss(sigDecoder_x, sig)
			KLD_loss = (0.5 * torch.sum(1 + logvar_x - mu_x.pow(2) - logvar_x.exp())) + (0.5 * torch.sum(1 + logvar_sig - mu_sig.pow(2) - logvar_sig.exp()))
			distributed_loss = torch.sqrt(torch.sum((mu_x - mu_sig) ** 2, dim=1) + torch.sum((torch.sqrt(logvar_x.exp()) - torch.sqrt(logvar_sig.exp())) ** 2, dim=1))
			distributed_loss = distributed_loss.sum()

			self.optimizer.zero_grad()

			loss = vae_reconstruction_loss - self.beta*KLD_loss
			if self.delta > 0:
				loss += distributed_loss*self.delta
			if self.gamma > 0:
				loss += cross_reconstruction_loss*self.gamma

			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			trainbar.set_description('l:%.3f' %(train_loss/(batch_idx+1)))
		
		#print("vae %f, da %f, ca %f"%(vae,da,ca))
		print(train_loss/(batch_idx+1))
		
		if epoch%100==0:
			name = "models/checkpoint_cada.pth"
			torch.save({
				'epoch':epoch,
				'model_encoder_state_dict':self.model_encoder.state_dict(),
				'model_decoder_state_dict':self.model_decoder.state_dict(),				
				'optimizer_state_dict':optimizer.state_dict(),
				'loss':loss,
				}, name)
		


	def extract_features(self, split):
		'''

		This function is used to extract the features from the encoder network

		'''
		features_img = []
		features_att = []
		gt = []
		self.model_encoder.eval()

		if split == 'trainval':
			num_classes = self.trainval_set.__NumClasses__()
		if split == 'test_unseen':
			num_classes = self.test_set.__NumClasses__()
		if split == 'test_seen':
			num_classes = self.test_set_seen.__NumClasses__()

		for n in num_classes:
			# 50 latent features per seen classes and 100 latent features per unseen classes
			if split == 'trainval':
				num_features = 200
				indexs = self.trainval_set.__getLabels__(n)
				dataset = self.trainval_set
			if split == 'test_unseen':
				num_features = 400
				indexs = self.test_set.__getLabels__(n)
				dataset = self.test_set
			if split == 'test_seen':
				num_features = 200
				indexs = self.test_set_seen.__getLabels__(n)
				dataset = self.test_set_seen				

			index_of_classLabels = indexs.tolist()
			if len(index_of_classLabels) < num_features:
				j = num_features - len(index_of_classLabels)
				index_of_classLabels = index_of_classLabels[:j] + index_of_classLabels
			else:
				index_of_classLabels = index_of_classLabels[:num_features]
			#print(len(index_of_classLabels))
			x_list = []
			y_list = []
			sig_list = []
			with torch.no_grad():
				for i in index_of_classLabels:
					x,y,sig = dataset.__getitem__(i)
					x_list.append(torch.unsqueeze(x, dim=0))
					y_list.append(torch.unsqueeze(y, dim=0))
					sig_list.append(torch.unsqueeze(sig, dim=0))
				x_tensor = torch.cat(x_list)
				y_tensor = torch.cat(y_list)
				sig_tensor = torch.cat(sig_list)

				z_imgs, z_atts, mu_imgs, logvar_imgs, mu_att, logvar_att = self.model_encoder(x_tensor, sig_tensor)
				features_img.extend(torch.unsqueeze(z_imgs.detach(), dim=0)) #make it a torch tensor
				features_att.extend(torch.unsqueeze(z_atts.detach(), dim=0))
				gt.extend(y_tensor)

		return features_img, features_att, gt, num_classes



	##################### FEATURE EXTRCTION #######################
	def prepare_data_classifier(self,batch_size):
		print(20*'-')
		print("Preparing dataset for the classifier..")
		trainval_features_img, trainval_features_att, trainval_labels, self.trainval_target_classes = self.extract_features('trainval')
		print(">> Extraction of trainval features is complete!")
		test_features_img_unseen, test_features_att_unseen, test_labels_unseen, self.test_unseen_target_classes = self.extract_features('test_unseen')
		print(">> Extraction of test_unseen features is complete!")
		test_features_img_seen, test_features_att_seen, test_labels_seen, self.test_seen_target_classes = self.extract_features('test_seen')
		print(">> Extraction of test_seen features is complete!")

		self.cls_data_trainval = classifier_dataloader(trainval_features_img, trainval_features_att, trainval_labels)
		self.cls_data_test_unseen = classifier_dataloader(test_features_img_unseen, test_features_att_unseen, test_labels_unseen)
		self.cls_data_test_seen = classifier_dataloader(test_features_img_seen, test_features_att_seen, test_labels_seen)

		self.cls_trainloader = data.DataLoader(self.cls_data_trainval, batch_size=batch_size, shuffle=True)
		self.cls_testloader_unseen = data.DataLoader(self.cls_data_test_unseen, batch_size=batch_size, shuffle=False)
		self.cls_testloader_seen = data.DataLoader(self.cls_data_test_seen, batch_size=batch_size, shuffle=False)

	##################### TRAINING THE CLASSIFIER #######################
	def train_classifier(self,epochs):
		best_H = -1
		best_seen = 0
		best_unseen = 0
		############## TRAINING ####################
		for epoch in range(1, epochs+1):
			print("Training: Epoch - ", epoch)
			self.classifier.train()
			trainbar_cls = tqdm(self.cls_trainloader)
			train_loss = 0
			for batch_idx,(x, sig, y) in enumerate(trainbar_cls):
				output = self.classifier(x)
				loss = self.lossfunction_classifier(output,y)
				self.cls_optimizer.zero_grad()
				loss.backward()
				self.cls_optimizer.step()
				train_loss += loss.item()
				trainbar_cls.set_description('l:%.3f' %(train_loss/(batch_idx+1)))

			########## VALIDATION ##################
			#accu_unseen = 0
			#accu_seen = 0
			def val_gzsl(testbar_cls):
				with torch.no_grad():
					self.classifier.eval()
					print("**Validation**")
					preds = []
					target = []
					for batch_idx, (x, sig, y) in enumerate(testbar_cls):
						output = self.classifier(x)
						output_data = torch.argmax(output.data,1)
						preds.append(output_data)
						target.append(y)	
					predictions = torch.cat(preds)
					targets = torch.cat(target)
					return predictions, targets

			testbar_cls_unseen = tqdm(self.cls_testloader_unseen)
			testbar_cls_seen = tqdm(self.cls_testloader_seen)

			preds_unseen, target_unseen = val_gzsl(testbar_cls_unseen)
			preds_seen, target_seen = val_gzsl(testbar_cls_seen)

			############### ACCURACY METRIC ##################
			def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
				# per_class_accuracies = torch.zeros(target_classes.shape[0]).float().to(self.device)
				# predicted_label = predicted_label.to(self.device)
				# for i in range(target_classes.shape[0]):
				# 	is_class = test_label==target_classes[i]
				# 	per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
				# return per_class_accuracies.mean()
				cm = confusion_matrix(test_label, predicted_label)
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				avg = sum(cm.diagonal())/len(target_classes)
				return avg

			#adding 1 to index classes from 1-200
			accu_unseen = compute_per_class_acc_gzsl(target_unseen, preds_unseen+1, self.test_unseen_target_classes)
			accu_seen = compute_per_class_acc_gzsl(target_seen, preds_seen+1, self.test_seen_target_classes)

			if (accu_seen+accu_unseen)>0:
				H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
			else:
				H = 0

			if H > best_H:

				best_seen = accu_seen
				best_unseen = accu_unseen
				best_H = H

			print(20*'-')
			print('Epoch:', epoch)
			print('u, s, h =%.4f,%.4f,%.4f'%(best_unseen,best_seen,best_H))
			print(20*'-')
				
		return best_seen, best_unseen, best_H



if __name__=='__main__':
	model = Gzsl_vae()
	epochs=100
	for epoch in range(1, epochs + 1):
		print("epoch:", epoch)
		model.train(epoch)

	#CLASSIFIER
	nepochs = 20
	model.prepare_data_classifier(batch_size=100)
	s, u, h = model.train_classifier(nepochs)



'''
To check:
1. Are the features being extracted properly?
2. Is the evaluation metric correct?
3. Is the classifier correct?

'''
	

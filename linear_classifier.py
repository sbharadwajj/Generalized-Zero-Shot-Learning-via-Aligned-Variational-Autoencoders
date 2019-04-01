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
import math

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
parser.add_argument('--model_path', type=str, default='../models/checkpoint_cada.pth',
					help='path of pretrained model')
parser.add_argument('--device', type=str, default='cpu',
					help='cuda or cpu')
parser.add_argument('--pretrained', default=False, action="store_true" , help="Load pretrained weights")
args = parser.parse_args()


class Gzsl_vae():
	"""docstring for Gzsl_vae"""
	def __init__(self,args):
		self.device = torch.device(args.device)

		######################## LOAD DATA #############################
		self.scalar = MinMaxScaler()
		self.trainval_set = dataloader(transform=self.scalar,root=args.dataset_path,split='trainval', device=self.device)
		#train_set = dataloader(root=args.dataset_path,split='train', device=self.device)
		self.test_set_unseen = dataloader(transform=self.scalar,root=args.dataset_path,split='test_unseen', device=self.device)
		self.test_set_seen = dataloader(transform=self.scalar,root=args.dataset_path,split='test_seen', device=self.device)
		#val_set = dataloader(root=args.dataset_path,split='val', device=self.device)		

		self.trainloader = data.DataLoader(self.trainval_set, batch_size=args.batch_size, shuffle=True)
		#self.testloader_unseen = data.DataLoader(self.test_set_unseen, batch_size=args.batch_size, shuffle=False) #for val	
		#self.testloader_seen = data.DataLoader(self.test_set_seen, batch_size=args.batch_size, shuffle=False) #for val

		self.input_dim = self.trainval_set.__getlen__()
		self.atts_dim = self.trainval_set.__get_attlen__()
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

		################### LOAD PRETRAINED MODEL ########################
		if args.pretrained:
			if args.model_path == '':
				print("Please provide the path of the pretrained model.")
			else:
				checkpoint = torch.load(args.model_path)
				self.model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
				self.model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
				print(">> Pretrained model loaded!")

		########## LOSS ############
		self.l1_loss = nn.L1Loss(reduction='sum')
		self.lossfunction_classifier =  nn.NLLLoss()


		######### Hyper-params #######
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
			name = "models/checkpoint_cada_AWA1.pth"
			torch.save({
				'epoch':epoch,
				'model_encoder_state_dict':self.model_encoder.state_dict(),
				'model_decoder_state_dict':self.model_decoder.state_dict(),				
				'optimizer_state_dict':self.optimizer.state_dict(),
				'loss':loss,
				}, name)
		


	##################### FEATURE EXTRCTION #######################
	def extract_features(self,params):
		print(20*'-')
		print("Preparing dataset for the classifier..")

		img_seen_feats = params['img_seen']
		img_unseen_feats = params['img_unseen']
		att_seen_feats = params['att_seen']
		att_unseen_feats = params['att_unseen']

		seen_classes = self.trainval_set.__NumClasses__()
		unseen_classes = self.test_set_unseen.__NumClasses__()

		#atts for unseen classes
		attribute_vector_unseen, labels_unseen = self.test_set_unseen.__attributeVector__()

		#for trainval features:
		features_seen = []
		labels_seen = []
		for n in seen_classes:
			perclass_feats = self.trainval_set.__get_perclass_feats__(n)
			repeat_factor = math.ceil(img_seen_feats/perclass_feats.shape[0])
			perclass_X = np.repeat(perclass_feats, repeat_factor, axis=0)
			perclass_labels = torch.from_numpy(np.repeat(n, img_seen_feats, axis=0)).long()
			seen_feats = perclass_X[:img_seen_feats].float()
			# if seen_feats.shape[0] < 200:
			# 	print(n,"-------", seen_feats.shape)
			features_seen.append(seen_feats)
			labels_seen.append(perclass_labels)

		tensor_seen_features = torch.cat(features_seen)
		tensor_seen_feats_labels = torch.cat(labels_seen)
		tensor_unseen_attributes = torch.from_numpy(np.repeat(attribute_vector_unseen,att_unseen_feats,axis=0)).float()
		tensor_unseen_labels = torch.from_numpy(np.repeat(labels_unseen,att_unseen_feats,axis=0)).long()

		test_unseen_X, test_unseen_Y = self.test_set_unseen.__Test_Features_Labels__()
		test_seen_X, test_seen_Y = self.test_set_seen.__Test_Features_Labels__()

		with torch.no_grad():
			z_img, z_att, mu_x, logvar_x, mu_att, logvar_att = self.model_encoder(tensor_seen_features, tensor_unseen_attributes)
			z_unseen_test_img, z_unseen_test_att, mu_x_unseen, logvar_x, mu_att, logvar_att = self.model_encoder(test_unseen_X, tensor_unseen_attributes)
			z_seen_test_img, z_unseen_test_att, mu_x_seen, logvar_x, mu_att, logvar_att = self.model_encoder(test_seen_X, tensor_unseen_attributes)

			train_features = torch.cat((z_att,z_img))
			train_labels = torch.cat((tensor_unseen_labels, tensor_seen_feats_labels))

		test_unseen_Y = torch.squeeze(test_unseen_Y)
		test_seen_Y = torch.squeeze(test_seen_Y)

		print(">> Extraction of trainval, test seen, and test unseen features are complete!")
		print(train_features.shape, train_labels.shape)
		#return train_features, train_labels, z_unseen_test_img, test_unseen_Y, z_seen_test_img, test_seen_Y
		return train_features, train_labels, mu_x_unseen, test_unseen_Y, mu_x_seen, test_seen_Y


	##################### TRAINING THE CLASSIFIER #######################
	def train_classifier(self,epochs):
		train_features, train_labels, test_unseen_features, test_unseen_labels, test_seen_features, test_seen_labels = self.extract_features(params)

		self.cls_trainData = classifier_dataloader(features_img=train_features, labels=train_labels, device=self.device)
		self.cls_trainloader = data.DataLoader(self.cls_trainData, batch_size=32, shuffle=True)

		self.cls_test_unseen = classifier_dataloader(features_img=test_unseen_features, labels=test_unseen_labels, device=self.device)
		self.cls_test_unseenLoader = data.DataLoader(self.cls_test_unseen, batch_size=32, shuffle=False)
		self.test_unseen_target_classes = self.cls_test_unseen.__targetClasses__()		

		self.cls_test_seen = classifier_dataloader(features_img=test_seen_features, labels=test_seen_labels, device=self.device)
		self.cls_test_seenLoader = data.DataLoader(self.cls_test_seen, batch_size=32, shuffle=False)
		self.test_seen_target_classes = self.cls_test_seen.__targetClasses__()


		best_H = -1
		best_seen = 0
		best_unseen = 0
		############## TRAINING ####################
		for epoch in range(1, epochs+1):
			print("Training: Epoch - ", epoch)
			self.classifier.train()
			trainbar_cls = tqdm(self.cls_trainloader)
			train_loss = 0
			for batch_idx,(x, y) in enumerate(trainbar_cls):
				output = self.classifier(x)
				loss = self.lossfunction_classifier(output,y)
				self.cls_optimizer.zero_grad()
				loss.backward()
				self.cls_optimizer.step()
				train_loss += loss.item()
				trainbar_cls.set_description('l:%.3f' %(train_loss/(batch_idx+1)))

			########## VALIDATION ##################
			accu_unseen = 0
			accu_seen = 0
			def val_gzsl(testbar_cls):
				with torch.no_grad():
					self.classifier.eval()
					print("**Validation**")
					preds = []
					target = []
					for batch_idx, (x, y) in enumerate(testbar_cls):
						output = self.classifier(x)
						output_data = torch.argmax(output.data,1)
						preds.append(output_data)
						target.append(y)	
					predictions = torch.cat(preds)
					targets = torch.cat(target)
					return predictions, targets

			testbar_cls_unseen = tqdm(self.cls_test_unseenLoader)
			testbar_cls_seen = tqdm(self.cls_test_seenLoader)

			preds_unseen, target_unseen = val_gzsl(testbar_cls_unseen)
			preds_seen, target_seen = val_gzsl(testbar_cls_seen)

			########## ACCURACY METRIC ##################
			def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
				per_class_accuracies = torch.zeros(target_classes.shape[0]).float().to(self.device)
				predicted_label = predicted_label.to(self.device)
				for i in range(target_classes.shape[0]):
					is_class = test_label==target_classes[i]
					per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
				return per_class_accuracies.mean()

			##################################
			'''
			For NLLL loss the labels are 
			mapped from 0-n, map them back to 1-n 
			for calculating accuracies.
			'''
			target_unseen = target_unseen + 1
			preds_unseen = preds_unseen + 1
			target_seen = target_seen + 1
			preds_seen = preds_seen + 1
			##################################


			accu_unseen = compute_per_class_acc_gzsl(target_unseen, preds_unseen, self.test_unseen_target_classes)
			accu_seen = compute_per_class_acc_gzsl(target_seen, preds_seen, self.test_seen_target_classes)

			if (accu_seen+accu_unseen)>0:
				H = (2*accu_seen*accu_unseen) / (accu_seen+accu_unseen)
			else:
				H = 0

			if H > best_H:

				best_seen = accu_seen
				best_unseen = accu_unseen
				best_H = H

			print(20*'-')
			print('Epoch:', epoch)
			print('u, s, h =%.4f,%.4f,%.4f'%(best_unseen,best_seen,best_H))
			print('u, s, h =%.4f,%.4f,%.4f'%(accu_unseen,accu_seen,H))
			print(20*'-')
				
		return best_seen, best_unseen, best_H



if __name__=='__main__':
	model = Gzsl_vae(args)
	if not args.pretrained:
		epochs=100
		for epoch in range(1, epochs + 1):
			print("epoch:", epoch)
			model.train(epoch)
	else:
		#CLASSIFIER
		params = {'img_seen':200,
				'img_unseen':0,
				'att_seen':0,
				'att_unseen':400}

		nepochs = 40
		s, u, h = model.train_classifier(nepochs)



'''
To complete:
1. classifier not updating after 2nd epoch 


'''
	

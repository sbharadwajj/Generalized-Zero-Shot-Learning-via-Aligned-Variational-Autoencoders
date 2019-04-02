import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.utils.data as data
import scipy.io
import numpy as np
from tqdm import tqdm

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.bias.data.fill_(0)

		nn.init.xavier_uniform_(m.weight,gain=0.5)


class encoder_cada(nn.Module):
	"""
	This is the encoder class which consists of the
	encoder for features and the attributes.

	features: x
	attributes: att
	"""
	def __init__(self, input_dim=2048, atts_dim=312, z=64 ):
		super(encoder_cada, self).__init__()
		self.encoder_x = nn.Sequential(nn.Linear(input_dim, 1560), nn.ReLU())
		self.mu_x = nn.Linear(1560, z)
		self.logvar_x = nn.Linear(1560, z)		
		
		self.encoder_att = nn.Sequential(nn.Linear(atts_dim, 1450), nn.ReLU())
		self.mu_att = nn.Linear(1450, z)
		self.logvar_att = nn.Linear(1450, z)

		self.apply(weights_init)

	def reparameterize(self, mu, logvar):
		# std = torch.exp(logvar) 
		# eps = torch.randn_like(std) # mean 0, std
		# return eps.mul(std).add_(mu)
		sigma = torch.exp(logvar)
		eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1)
		eps  = eps.expand(sigma.size())
		return mu + sigma*eps

	def forward(self, x, att):
		x = self.encoder_x(x)
		mu_x = self.mu_x(x)
		logvar_x = self.logvar_x(x)
		z_x = self.reparameterize(mu_x, logvar_x)

		att = self.encoder_att(att)
		mu_att = self.mu_att(att)
		logvar_att = self.logvar_att(att)
		z_att = self.reparameterize(mu_att,logvar_att)

		return z_x, z_att, mu_x, logvar_x, mu_att, logvar_att

class decoder_cada(nn.Module):
	"""docstring for decoder_cada"""
	def __init__(self, input_dim=2048, atts_dim=312, z=64):
		super(decoder_cada, self).__init__()
		self.decoder_x = nn.Sequential(nn.Linear(z, 1660), nn.ReLU(), nn.Linear(1660, input_dim))
		self.decoder_att = nn.Sequential(nn.Linear(z, 665),nn.ReLU(), nn.Linear(665, atts_dim))

		self.apply(weights_init)


	def forward(self, z_x, z_att):
		recon_x = self.decoder_x(z_x)
		recon_att = self.decoder_att(z_att)				
		
		att_recon_x = self.decoder_att(z_x)
		x_recon_att = self.decoder_x(z_att)

		return recon_x, recon_att, att_recon_x, x_recon_att


class Classifier(nn.Module):
	def __init__(self, input_dim, num_class):
		super(Classifier, self).__init__()
		self.fc = nn.Linear(input_dim,num_class)
		self.softmax = nn.LogSoftmax(dim=1)

		self.apply(weights_init)

	def forward(self, features):
		x = self.softmax(self.fc(features))

		return x


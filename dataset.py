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
from sklearn.preprocessing import MinMaxScaler

class dataloader(Dataset):
	"""docstring for CUB"""
	def __init__(self, root='/content/drive/My Drive/computer_vision/datasets/CUB/', split='train', device='cpu',transform=None):
		path_features = root + 'res101.mat'
		path_att_splits = root + 'att_splits.mat'
		self.res101 = scipy.io.loadmat(path_features)
		att_splits = scipy.io.loadmat(path_att_splits)

		self.scaler = MinMaxScaler()

		self.labels, self.feats, self.sig = self.get_data(att_splits, split)
		assert len(self.labels) == len(self.feats) == len(self.sig)
		if len(self.feats) == 0:
			raise(RuntimeError("Found zero feats in the directory: "+ root))

		self.feats_ = torch.from_numpy(self.feats).float().to(device)
		self.labels_ = torch.from_numpy(self.labels).float().to(device)

		# ONE HOT ENCODED
		self.sig_ = torch.from_numpy(self.sig).float().to(device)

	def __getitem__(self, index):
		#index = np.random.randint(1,50)
		x = self.feats_[index,:]
		sig = self.sig_[index,:]
		y = self.labels_[index]
		return x,y,sig

	def __getLabels__(self, index):
		if index in np.unique(self.labels_):
			idx = np.where(self.labels_==index)
			return idx[0]

	def __NumClasses__(self):
		return np.unique(self.labels_)

	def __get_attlen__(self):
		len_sig = self.sig.shape[1]
		return len_sig

	def __getlen__(self):
		len_feats = self.feats.shape[1]
		return len_feats

	def __totalClasses__(self):
		return len(np.unique(self.res101['labels']).tolist())


	def check_unique_labels(self, labels, att_splits):
		trainval_loc = 'trainval_loc'
		train_loc = 'train_loc'
		val_loc = 'val_loc'
		test_loc = 'test_unseen_loc'

		self.labels_train = labels[np.squeeze(att_splits[train_loc]-1)]
		self.labels_val = labels[np.squeeze(att_splits[val_loc]-1)]
		self.labels_trainval = labels[np.squeeze(att_splits[trainval_loc]-1)]
		self.labels_test = labels[np.squeeze(att_splits[test_loc]-1)]

		self.train_labels_seen = np.unique(self.labels_train)
		self.val_labels_unseen = np.unique(self.labels_val)
		self.trainval_labels_seen = np.unique(self.labels_trainval)
		self.test_labels_unseen = np.unique(self.labels_test)   

		print("Number of overlapping classes between train and val:",
			len(set(self.train_labels_seen).intersection(set(self.val_labels_unseen))))
		print("Number of overlapping classes between trainval and test:",
			len(set(self.trainval_labels_seen).intersection(set(self.test_labels_unseen))))
		
	def __len__(self):
		return self.feats.shape[0]
		
	def get_data(self, att_splits, split):
		labels = self.res101['labels']
		X_features = self.res101['features']
		signature = att_splits['att']
		
		self.check_unique_labels(labels, att_splits)
		if split == 'trainval':
			loc = 'trainval_loc'
		elif split == 'train':
			loc = 'train_loc'
		elif split == 'val':
			loc = 'val_loc'
		elif split == 'test_seen':
			loc = 'test_seen_loc'
		else:
			loc = 'test_unseen_loc'

		labels_loc = labels[np.squeeze(att_splits[loc]-1)]
		feat_vec = np.transpose(X_features[:,np.squeeze(att_splits[loc]-1)])
		
		unique_labels = np.unique(labels_loc)
		sig_vec = np.zeros((labels_loc.shape[0],signature.shape[0]))
		labels_list = np.squeeze(labels_loc).tolist()
		for i, idx in enumerate(labels_list):
		  sig_vec[i,:] = signature[:,idx-1]
		
		self.scaler.fit_transform(feat_vec)

		return labels_loc, sig_vec, feat_vec
		
class classifier_dataloader(Dataset):
	"""docstring for classifier_dataloader"""
	def __init__(self, features_img, features_att, labels):
		self.labels = labels
		self.feats = torch.cat(features_img)
		self.atts = torch.cat(features_att)
		self.y = torch.cat(labels).long()

	def __getitem__(self,index):
		X = self.feats[index, :]
		atts = self.atts[index, :]
		y = self.y[index]
		return X, atts, y

	def __len__(self):
		return len(np.unique(self.labels).tolist())

	def __targetClasses__(self):
		return np.unique(labels)

		
		

# if __name__ == "__main__":
#   dataset = DatasetLoader(split='train')
#   x,y,sig = dataset.__getitem__(5)
#   #len = dataset.__getlen__()
#   #print(len)
#   #sig = dataset.__get_attlen__()
#   #print(sig)
#   trainloader = data.DataLoader(dataset, batch_size=125, shuffle=True)
#   tbar = tqdm(trainloader)
#   #for batch_idx, (x,y,sig) in enumerate(trainloader):
#     #print(batch_idx)
#   model = CADA_VAE()
#   recon_x, recon_sig, mu_z, mu_sig, sigDecoder_x, xDecoder_sig,logvar_x, logvar_sig = model(x, sig)
#   print(recon_sig.shape)

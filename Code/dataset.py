import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchaudio

import glob
import pickle as pkl
import numpy as np
from typing import List

#from debug_flags import _control_flags
#from metrics import _my_istft

class ToTensor(object):
	r"""Convert ndarrays in sample to Tensors."""

	def __call__(self, x):
		return torch.from_numpy(x)

class IEEE_Dataset(Dataset):
	# supports for train and dev
	# (SISO, MISO, MIMO)
	# net_tgt: ["PSM", "IRM", "CSM"]
	# net_inp: ["LOG_PS", "CSM"]
	# (ref_mic_idx: [0, 1, -1])
	def __init__(self, feat_filedir:str, mean_var_norm_file_path: str, net_type: str, net_inp:str, net_tgt:str, ref_mic_idx: int, mean_var_norm_flag:bool = True, train_flag: bool = False, size:int =None):
		self.feat_filedir = feat_filedir
		self.files_list = sorted(glob.glob(self.feat_filedir + '/*.pkl'))
		with open(mean_var_norm_file_path, "rb") as f_pkl: 
			self.mean_var_norm_dict = pkl.load(f_pkl) 
		self.to_tensor = ToTensor()
		self.net_type = net_type
		self.net_inp = net_inp
		self.net_tgt = net_tgt
		self.ref_mic_idx = ref_mic_idx
		self.max_seq_len = 407 #dev tr: 385
		self.mean_var_norm_flag = mean_var_norm_flag
		self.train_flag = train_flag
		self.size = size

	def __len__(self):
		return ( len(self.files_list)*2 if self.train_flag and ("SISO"==self.net_type) else len(self.files_list) ) if self.size is None else self.size
	
	def get_real_imag(self, x):
		#assuming x:(ch, T, F) complex
		#output: (2*ch, T, F) (r,i,r,i)
		_real = np.real(x)
		_imag = np.imag(x)
		num_ch = x.shape[0]
	
		return np.concatenate( [ np.stack((_real[ch,:,:], _imag[ch,:,:]),axis=0) for ch in range(num_ch) ] , axis=0)

	def miso_mimo_mean_var_norm(self, x, mean, var):
		#assuming x.shape (num_ch*2, T, F) 
		return (x- np.expand_dims(mean, axis=1))/np.expand_dims(var, axis=1)

	def siso_mean_var_norm(self, x, mean, var):
		#assuming x.shape (T,F)
		return (x-np.expand_dims(mean,axis=0))/np.expand_dims(var, axis=0)
	
	def _get_item_test_siso(self, _dict):
		#"supports only csm for now can be modified for masking"		
		mix_cs = _dict["mix"]
		doa = _dict["doa_degrees"]
		
		##aachen mixture code (picking 8 cm )
		if 8==mix_cs.shape[0]:
			mix_cs = mix_cs[3:5,:,:]

		if "LOG_PS" == self.net_inp:
			log_mix_ps = _dict["log_mix_ps"] #real
			if 8==log_mix_ps.shape[0]:
				log_mix_ps = log_mix_ps[3:5,:,:]
			unpadded_seq_len = log_mix_ps.shape[1]
			if self.mean_var_norm_flag:
				log_mix_ps_norm = self.siso_mean_var_norm(log_mix_ps, np.expand_dims(self.mean_var_norm_dict["mean"], axis=0), np.expand_dims(self.mean_var_norm_dict["std"], axis=0))
			else:
				log_mix_ps_norm = log_mix_ps
		
		if "PSM"==self.net_tgt:
			tgt = _dict["psm"]
		elif "IRM"==self.net_tgt:
			tgt = _dict["irm"]
		elif "CSM"==self.net_tgt:
			tgt = _dict["dp_sph"]
			tgt_ri = self.get_real_imag(tgt)	

		if 8==tgt.shape[0]:
			tgt = tgt[3:5,:,:]

		if self.mean_var_norm_flag and self.net_inp=="CSM":
			mix_cs_norm = self.miso_mimo_mean_var_norm(mix_cs, self.mean_var_norm_dict["mean_cs"], self.mean_var_norm_dict["std_cs"])
		else:
			mix_cs_norm = mix_cs

		#reshape (2,T,F) complex to (4, T,F) real,imag
		mix_ri = self.get_real_imag(mix_cs_norm)
		
		if "LOG_PS" == self.net_inp:
			return log_mix_ps_norm, tgt, torch.tensor([unpadded_seq_len, unpadded_seq_len]) , doa, mix_ri
		else:
			return mix_ri, tgt_ri, unpadded_seq_len, doa, _
	
	def _get_item_miso_mimo(self, _dict):
		#"supports only csm for now can be modified for masking"		
		mix_cs = _dict["mix"]
		dp_sph = _dict["dp_sph"]
		doa = _dict["doa_degrees"]

		if self.mean_var_norm_flag:
			mix_cs_norm = self.miso_mimo_mean_var_norm(mix_cs, self.mean_var_norm_dict["mean_cs"], self.mean_var_norm_dict["std_cs"])
		else:
			mix_cs_norm = mix_cs

		if self.ref_mic_idx != -1:
			tgt = dp_sph[[self.ref_mic_idx],:,:]
		else:
			tgt = dp_sph

		#reshape (2,T,F) complex to (4, T,F) real,imag

		mix_ri = self.get_real_imag(mix_cs_norm)
		tgt_ri = self.get_real_imag(tgt)

		return mix_ri, tgt_ri, doa

	def _get_item_siso(self, _dict, ch):

		if "LOG_PS"==self.net_inp:
			log_mix_ps = _dict["log_mix_ps"] #real
			log_mix_ps_ch = log_mix_ps[ch,:,:]

			if self.mean_var_norm_flag:
				log_mix_ps_norm = self.siso_mean_var_norm(log_mix_ps_ch, self.mean_var_norm_dict["mean"], self.mean_var_norm_dict["std"])
				inp = log_mix_ps_norm
			else:
				inp = log_mix_ps_ch

		else:
			mix_cs = _dict["mix"] #complex
			mix_cs_ch = mix_cs[ch,:,:]
			if self.mean_var_norm_flag:
				mix_cs_norm = self.siso_mean_var_norm(mix_cs_ch, self.mean_var_norm_dict["mean_cs"][ch,:], self.mean_var_norm_dict["std_cs"][ch,:])
			else:
				mix_cs_norm = mix_cs_ch
			inp = self.get_real_imag(np.expand_dims(mix_cs_norm, axis=0))
		
		if "PSM"==self.net_tgt:
			tgt = _dict["psm"]
			tgt = tgt[ch,:,:]
		elif "IRM"==self.net_tgt:
			tgt = _dict["irm"]
			tgt = tgt[ch,:,:]
		elif "CSM"==self.net_tgt:
			tgt = _dict["dp_sph"]	
			tgt = tgt[ch,:,:]
			tgt = self.get_real_imag(np.expand_dims(tgt, axis=0))

		doa = _dict["doa_degrees"]

		return inp, tgt, doa
	
	def __getitem__(self, index):
		'''
			Assumption: each file has 2 channel data
		'''
		index_file = index

		if self.train_flag:
			if self.net_type=="SISO":
				index_file = int(np.floor(index/2))
				_ch = index %2

		filename = self.files_list[index_file]

		with open(filename, "rb") as f_pkl:
			_dict = pkl.load(f_pkl)

		
		if self.net_type=="SISO":

			if self.train_flag:
				if self.net_inp=="CSM" and self.net_tgt=="CSM":
					mix_cs, dp_sph_tgt, doa = self._get_item_siso(_dict, _ch)
					unpadded_seq_len = mix_cs.shape[1]
					mix_cs = self.pad_seq_miso_mimo(mix_cs)
					dp_sph_tgt = self.pad_seq_miso_mimo(dp_sph_tgt)

					return self.to_tensor(mix_cs).to(torch.float32), self.to_tensor(dp_sph_tgt).to(torch.float32), unpadded_seq_len, doa
				else:
					mix_log_ps, mix_psm, doa = self._get_item_siso(_dict, _ch)
					unpadded_seq_len = mix_log_ps.shape[0]
					mix_log_ps = self.pad_seq_siso(mix_log_ps)
					mix_psm = self.pad_seq_siso(mix_psm)
					
					mix_cs = self.get_real_imag(_dict["mix"][[_ch],:,:])
					mix_cs = self.pad_seq_miso_mimo(mix_cs)
					
					return self.to_tensor(mix_log_ps).to(torch.float32), self.to_tensor(mix_psm).to(torch.float32), unpadded_seq_len, doa, self.to_tensor(mix_cs).to(torch.float32)
			else:
				mix_log_ps, tgt, unpadded_seq_len, doa, mix_cs = self._get_item_test_siso(_dict)
				return self.to_tensor(mix_log_ps).to(torch.float32), self.to_tensor(tgt).to(torch.float32), unpadded_seq_len, doa, self.to_tensor(mix_cs).to(torch.float32)

		else:
			mix_cs, dp_sph_tgt, doa = self._get_item_miso_mimo(_dict)
			unpadded_seq_len = mix_cs.shape[1]
			mix_cs = self.pad_seq_miso_mimo(mix_cs)
			dp_sph_tgt = self.pad_seq_miso_mimo(dp_sph_tgt)	

			return self.to_tensor(mix_cs).to(torch.float32), self.to_tensor(dp_sph_tgt).to(torch.float32), unpadded_seq_len, doa

	def pad_seq_siso(self, x):
		(n_frms, n_freq) = x.shape
		req_dtype = x.dtype
		if n_frms < self.max_seq_len:
			req_pad = self.max_seq_len - n_frms
			_pad = np.zeros((req_pad, n_freq), dtype=req_dtype)
			x = np.concatenate((x, _pad), axis=0)
		return x

	def pad_seq_miso_mimo(self, x):
		(num_ch, n_frms, n_freq) = x.shape
		req_dtype = x.dtype
		if n_frms < self.max_seq_len:
			req_pad = self.max_seq_len - n_frms
			_pad = np.zeros((num_ch, req_pad, n_freq), dtype=req_dtype)
			x = np.concatenate((x, _pad), axis=1)
		return x
	
	def get_ip_seq_len(self) -> List:
		_ch = 0
		seq_len_lst = []
		for _filename in self.files_list:
			with open(_filename, "rb") as f_pkl:
				_dict = pkl.load(f_pkl)

			log_ps = _dict["log_mix_ps"][_ch]
			seq_len_lst.append(log_ps.shape[0])

		max_seq_len = max(seq_len_lst)
		return seq_len_lst, max_seq_len


if __name__== "__main__":

	feat_path = '/scratch/bbje/battula12/Datasets_paper/ZQ_Dataset/features_32_8_ms_20cm/devset'
	mv_norm_path = '/scratch/bbje/battula12/Datasets_paper/ZQ_Dataset/mean_variance_trainset.pkl'

	#("LOG_PS", "PSM") 
	#("CSM, "CSM")

	train_dataset = IEEE_Dataset(feat_path, mv_norm_path, "SISO", "CSM", "CSM", -1, True, True, 2)

	#seq_list, max_seq_list = train_dataset.get_ip_seq_len()
	#breakpoint()
	train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4)
	
	for idx, ex in enumerate(train_loader):
		(mix_inp, dp_sph_tgt, unpadded_seq_len, doa) = ex #mix_cs
		print(mix_inp.shape, dp_sph_tgt.shape, unpadded_seq_len, doa) #, mix_cs.shape
		#break




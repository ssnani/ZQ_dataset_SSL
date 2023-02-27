import torch
import torch.nn as nn
import torch.nn.functional as F

class ZQLSTM(nn.Module):
	def __init__(self, bidirectional: bool):
		super().__init__()
		
		self.inp_size = 257
		self.out_size = 257

		self.num_hid_lyrs = 2
		self.hid_size = 600
		self.bidirectional = bidirectional
		self.dropout = 0.1
		
		self.rnn = nn.LSTM( self.inp_size, self.hid_size, self.num_hid_lyrs, 
							batch_first=True, dropout=self.dropout,
							bidirectional=self.bidirectional
							) 

		self.rnn_out_size = self.hid_size*2 if self.bidirectional else self.hid_size

		self.mask_lin_lyr = nn.Linear(self.rnn_out_size, self.out_size)


	
	def forward(self, x, state):
		#assuming x.shape (B, T, *)
		rnn_out, _ = self.rnn(x, state)
		return torch.sigmoid(self.mask_lin_lyr(rnn_out))
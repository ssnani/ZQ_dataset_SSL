import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils.utils import numParams


class GLSTM(nn.Module):
    def __init__(self, hidden_size=1024, groups=4, bidirectional=True):
        super(GLSTM, self).__init__()
   
        hidden_size_t = hidden_size // groups

        hid_out = hidden_size_t//2 if bidirectional else hidden_size_t
        self.lstm_list1 = nn.ModuleList([nn.LSTM(hidden_size_t, hid_out, 1, batch_first=True, bidirectional=bidirectional) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList([nn.LSTM(hidden_size_t, hid_out, 1, batch_first=True, bidirectional=bidirectional) for i in range(groups)])

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
     
        self.groups = groups
     
    def forward(self, x, seq_len):
        out = x
        b_s, n_ch, pad_t, n_f = x.shape
        
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
    
        out = torch.chunk(out, self.groups, dim=-1)

        _packed_out = [ torch.nn.utils.rnn.pack_padded_sequence(out[i], seq_len.cpu(), batch_first=True, enforce_sorted=False) for i in range(self.groups) ]
        _packed_seq_out = [ self.lstm_list1[i](_packed_out[i])[0] for i in range(self.groups) ]
        out = [nn.utils.rnn.pad_packed_sequence(_packed_seq_out[i], batch_first=True, total_length = pad_t)[0] for i in range(self.groups) ]
        out = torch.stack(out, dim=-1)

        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)
    
        out = torch.chunk(out, self.groups, dim=-1)

        _packed_out = [ torch.nn.utils.rnn.pack_padded_sequence(out[i], seq_len.cpu(), batch_first=True, enforce_sorted=False) for i in range(self.groups) ]
        _packed_seq_out = [ self.lstm_list2[i](_packed_out[i])[0] for i in range(self.groups) ]
        out = [nn.utils.rnn.pad_packed_sequence(_packed_seq_out[i], batch_first=True, total_length = pad_t)[0] for i in range(self.groups) ]
        out = torch.cat(  out, dim=-1)

        out = self.ln2(out)
    
        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()
      
        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
   
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class DenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, grate):
        super(DenseConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, grate, (1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(in_channels+grate, grate, (1,3), padding=(0,1))
        self.conv3 = nn.Conv2d(in_channels+2*grate, grate, (1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(in_channels+3*grate, grate, (1,3), padding=(0,1))
        self.conv5 = GluConv2d(in_channels+4*grate, out_channels, kernel_size, padding=padding, stride=stride)
 
        self.bn1 = nn.BatchNorm2d(grate)
        self.bn2 = nn.BatchNorm2d(grate)
        self.bn3 = nn.BatchNorm2d(grate)
        self.bn4 = nn.BatchNorm2d(grate)
        
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()
  
    def forward(self, x):
        out = x
        out1 = self.elu1(self.bn1(self.conv1(out)))
        out = torch.cat([x, out1], dim=1)
        out2 = self.elu2(self.bn2(self.conv2(out)))
        out = torch.cat([x, out1, out2], dim=1)
        out3 = self.elu3(self.bn3(self.conv3(out)))
        out = torch.cat([x, out1, out2, out3], dim=1)
        out4 = self.elu4(self.bn4(self.conv4(out)))
        out = torch.cat([x, out1, out2, out3, out4], dim=1)
        out5 = self.conv5(out)
    
        out = out5
    
        return out


class DenseConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, grate):
        super(DenseConvTranspose2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, grate, (1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(in_channels+grate, grate, (1,3), padding=(0,1))
        self.conv3 = nn.Conv2d(in_channels+2*grate, grate, (1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(in_channels+3*grate, grate, (1,3), padding=(0,1))
        self.conv5 = GluConvTranspose2d(in_channels+4*grate, out_channels, kernel_size, padding=padding, stride=stride)

        self.bn1 = nn.BatchNorm2d(grate)
        self.bn2 = nn.BatchNorm2d(grate)
        self.bn3 = nn.BatchNorm2d(grate)
        self.bn4 = nn.BatchNorm2d(grate)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()

    def forward(self, x):
        out = x
        out1 = self.elu1(self.bn1(self.conv1(out)))
        out = torch.cat([x, out1], dim=1)
        out2 = self.elu2(self.bn2(self.conv2(out)))
        out = torch.cat([x, out1, out2], dim=1)
        out3 = self.elu3(self.bn3(self.conv3(out)))
        out = torch.cat([x, out1, out2, out3], dim=1)
        out4 = self.elu4(self.bn4(self.conv4(out)))
        out = torch.cat([x, out1, out2, out3, out4], dim=1)
        out5 = self.conv5(out)

        out = out5

        return out

            
class Net(nn.Module):
    def __init__(self, bidirectional, frame_size, net_type):
        super(Net, self).__init__()

        if net_type=="SISO":
            (inp_dim, out_dim) = ( 2, 2) #(r,i)
        elif net_type=="MISO":
            (inp_dim, out_dim) = ( 4, 2) #(r,i)
        else:
            (inp_dim, out_dim) = ( 4, 4) #(r,i)

        self.net_type = net_type
                
        self.conv1 = DenseConv2d(inp_dim, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2 = DenseConv2d(16, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3 = DenseConv2d(32, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4 = DenseConv2d(64, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv5 = DenseConv2d(128, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)

        if frame_size==320:
            red_factor = 5
        elif frame_size==512:
            red_factor = 8
        else:
            print("check red_factor")
        self.glstm = GLSTM(red_factor*256, 4, bidirectional)
        
        self.conv5_t = DenseConvTranspose2d(256+256, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4_t = DenseConvTranspose2d(256+128, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3_t = DenseConvTranspose2d(128+64, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2_t = DenseConvTranspose2d(64+32, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv1_t = DenseConvTranspose2d(32+16, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        
        fft_half = frame_size//2
        one_side_fft = fft_half+ 1
        
        self.fc1 = nn.Linear(fft_half*8, one_side_fft)
        self.fc2 = nn.Linear(fft_half*8, one_side_fft)

        if net_type=="MIMO":
            self.fc3 = nn.Linear(fft_half*8, one_side_fft)
            self.fc4 = nn.Linear(fft_half*8, one_side_fft)

        
        self.path1 = DenseConv2d(16, 16, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path2 = DenseConv2d(32, 32, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path3 = DenseConv2d(64, 64, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path4 = DenseConv2d(128, 128, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path5 = DenseConv2d(256, 256, (1,3), padding=(0,1), stride=(1,1), grate=8)

    def forward(self, x, seq_len):
        
        out = x
        e1 = self.conv1(out)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
        out = e5
        
        out = self.glstm(out, seq_len)
        
        out = torch.cat([out, self.path5(e5)], dim=1)
        d5 = torch.cat([self.conv5_t(out), self.path4(e4)], dim=1)
        d4 = torch.cat([self.conv4_t(d5), self.path3(e3)], dim=1)
        d3 = torch.cat([self.conv3_t(d4), self.path2(e2)], dim=1)
        d2 = torch.cat([self.conv2_t(d3), self.path1(e1)], dim=1)
        d1 = self.conv1_t(d2)
        #breakpoint()
        out1 = d1[:,:8].transpose(1,2).contiguous().view(d1.size(0), d1.size(2), -1).contiguous()
        out2 = d1[:,8:].transpose(1,2).contiguous().view(d1.size(0), d1.size(2), -1).contiguous()
       
        if self.net_type=="MIMO":
            out_r1 = self.fc1(out1)
            out_i1 = self.fc2(out2)

            out_r2 = self.fc3(out1)
            out_i2 = self.fc4(out2)

            out = torch.stack([out_r1, out_i1, out_r2, out_i2], dim=1)

        else:
            out1 = self.fc1(out1)
            out2 = self.fc2(out2)

            out = torch.stack([out1, out2], dim=1)
        return out

class MIMO_Net(nn.Module):
    def __init__(self, bidirectional, frame_size):
        super(MIMO_Net, self).__init__()
                
        self.conv1 = DenseConv2d(4, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2 = DenseConv2d(16, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3 = DenseConv2d(32, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4 = DenseConv2d(64, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv5 = DenseConv2d(128, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)

        if frame_size==320:
            red_factor = 5
        elif frame_size==512:
            red_factor = 8
        else:
            print("check red_factor")

        self.glstm = GLSTM(red_factor*256, 4, bidirectional)
        
        self.conv5_t = DenseConvTranspose2d(256+256, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4_t = DenseConvTranspose2d(256+128, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3_t = DenseConvTranspose2d(128+64, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2_t = DenseConvTranspose2d(64+32, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv1_t = DenseConvTranspose2d(32+16, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        
        fft_half = frame_size//2
        one_side_fft = fft_half+ 1

        self.fc1 = nn.Linear(fft_half*8, one_side_fft)
        self.fc2 = nn.Linear(fft_half*8, one_side_fft)

        self.fc3 = nn.Linear(fft_half*8, one_side_fft)
        self.fc4 = nn.Linear(fft_half*8, one_side_fft)
        
        self.path1 = DenseConv2d(16, 16, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path2 = DenseConv2d(32, 32, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path3 = DenseConv2d(64, 64, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path4 = DenseConv2d(128, 128, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path5 = DenseConv2d(256, 256, (1,3), padding=(0,1), stride=(1,1), grate=8)

    def forward(self, x, seq_len):
        
        out = x
        e1 = self.conv1(out)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
        out = e5
        
        out = self.glstm(out, seq_len)
        
        out = torch.cat([out, self.path5(e5)], dim=1)
        d5 = torch.cat([self.conv5_t(out), self.path4(e4)], dim=1)
        d4 = torch.cat([self.conv4_t(d5), self.path3(e3)], dim=1)
        d3 = torch.cat([self.conv3_t(d4), self.path2(e2)], dim=1)
        d2 = torch.cat([self.conv2_t(d3), self.path1(e1)], dim=1)
        d1 = self.conv1_t(d2)

        #breakpoint()
        out1 = d1[:,:8].transpose(1,2).contiguous().view(d1.size(0), d1.size(2), -1).contiguous()
        out2 = d1[:,8:].transpose(1,2).contiguous().view(d1.size(0), d1.size(2), -1).contiguous()

        out_r1 = self.fc1(out1)
        out_i1 = self.fc2(out2)

        #out_0 = torch.stack([out_r1, out_i1], dim=1)

        out_r2 = self.fc3(out1)
        out_i2 = self.fc4(out2)

        out = torch.stack([out_r1, out_i1, out_r2, out_i2], dim=1)

        return out
        
def test_model(bidirectional):
    feat = torch.randn(5, 2, 50, 257) #161
    net = Net(bidirectional, 512, "SISO")
    #print('n_params: {}'.format(numParams(net)))
    #breakpoint()
    #feat = feat.to('cuda:0')
    #net = net.to('cuda:0')
    import time
    start = time.time()
    est = net(feat, torch.tensor([45, 50, 35, 25, 30]).squeeze())
    end = time.time() - start 
    print('Time for {} -> {} is {} sec'.format(feat.shape, est.shape, end))



if __name__=="__main__":
    import sys
    bidirectional = sys.argv[1] == "True"
    test_model(bidirectional)
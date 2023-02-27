import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, labels, nframes):
        loss_mask = self.lossMask(labels, nframes)
        loss = ((outputs - labels) * loss_mask)**2
        loss = torch.sum(loss)
        loss_mask = torch.sum(loss_mask)
        #breakpoint()
        loss = loss/loss_mask
        return loss#, loss_mask

    def lossMask(self, labels, nframes):
        loss_mask = torch.zeros_like(labels).requires_grad_(False)
        for j, seq_len in enumerate(nframes):
            loss_mask.data[j,0:seq_len,:] += 1.0
        return loss_mask

class LossFunction(object):
    def __init__(self, loss_flag):
        self.eps = torch.finfo(torch.float32).eps
        self.loss_flag = loss_flag
    def __call__(self, est, lbl, nframes):
        batch_size, n_ch, n_frames, n_feats = est.shape

        est_mag_1 = torch.sqrt(est[:,0]**2 + est[:,1]**2 + self.eps)
        lbl_mag_1 = torch.sqrt(lbl[:,0]**2 + lbl[:,1]**2 + self.eps)

        if n_ch==4:
            est_mag_2 = torch.sqrt(est[:,2]**2 + est[:,3]**2 + self.eps)
            lbl_mag_2 = torch.sqrt(lbl[:,2]**2 + lbl[:,3]**2 + self.eps)

            est_mag = est_mag_1 + est_mag_2
            lbl_mag = lbl_mag_1 + lbl_mag_2
        else:
            est_mag = est_mag_1
            lbl_mag = lbl_mag_1

        loss_mask = self.lossMask(lbl, nframes)

        ri = torch.abs(est - lbl)*loss_mask
        mag = torch.abs(est_mag - lbl_mag)*loss_mask[:,0]

        n_act_frames = torch.sum(loss_mask[:,0])
        #print(n_act_frames)

        loss_ri = torch.sum(ri) / float(n_act_frames) #batch_size* n_frames * n_feats)
        loss_mag = torch.sum(mag) / float(n_act_frames) #batch_size* n_frames * n_feats)
        #loss = loss_ri + loss_mag

        #frame level metrics
        #loss_ri_frm = torch.sum(ri,dim=3)
        #loss_mag_frm = torch.sum(mag,dim=3)


        #tried loss_ph over loss_ri -> didn't work
        loss_ph = 0

        if self.loss_flag=="MISO_RI" or self.loss_flag=="SISO_RI":
            loss = loss_ri
        elif self.loss_flag=="MISO_RI_MAG" or self.loss_flag=="SISO_RI_MAG":
            loss = loss_ri + loss_mag
        else:
            print("Invalid loss flag")

        return loss, loss_ri, loss_mag, loss_ph

    def lossMask(self, labels, nframes):
        loss_mask = torch.zeros_like(labels).requires_grad_(False)
        for j, seq_len in enumerate(nframes):
            loss_mask.data[j,:,0:seq_len,:] += 1.0
        return loss_mask
    
class MIMO_LossFunction(object):
    def __init__(self, loss_flag):
        self.eps = torch.finfo(torch.float32).eps
        self.loss_flag = loss_flag

    def __call__(self, est, lbl, nframes):
        batch_size, n_ch, n_frames, n_feats = est.shape

        lbl_mag_1, lbl_ph_1, lbl_mag_2, lbl_ph_2 = self.get_mag_ph(lbl)
        est_mag_1, est_ph_1, est_mag_2, est_ph_2 = self.get_mag_ph(est)

        loss_mask = self.lossMask(lbl, nframes)

        #print(est_ph_1[0,:,:], est_ph_2[0,:,:])
        ri = torch.abs(est - lbl)*loss_mask
        mag = (torch.abs(est_mag_1 - lbl_mag_1) + torch.abs(est_mag_2 - lbl_mag_2))*loss_mask[:,0]
        
        n_act_frames = torch.sum(loss_mask[:,0])
        #print(n_act_frames)

        loss_ri = torch.sum(ri) / float(n_act_frames) #batch_size* n_frames * n_feats)
        loss_mag = torch.sum(mag) / float(n_act_frames) #batch_size* n_frames * n_feats)
        
        #loss = loss_ri + loss_mag 

        #ph_diff = torch.abs(torch.cos(lbl_ph_1-lbl_ph_2) - torch.cos(est_ph_1-est_ph_2))
        ph_diff = lbl_mag_1[:,:,1:n_feats-1]*lbl_mag_2[:,:,1:n_feats-1]*torch.abs((lbl_ph_1-lbl_ph_2) - (est_ph_1-est_ph_2))*loss_mask[:,0,:,1:n_feats-1]  #best
        #ph_diff = lbl_mag_1[:,:,1:n_feats-1]*lbl_mag_2[:,:,1:n_feats-1]*torch.cos((lbl_ph_1-lbl_ph_2) - (est_ph_1-est_ph_2))   
        #loss_ph_diff = torch.sum(ph_diff) / float(batch_size* n_frames * n_feats)
        #loss -= 0.01*loss_ph_diff #

        ## second order interaction 
        ## R1*R2 + I1*I2, I1R2-R1I2
        #est_r = est[:,0,:,:]*est[:,2,:,:] + est[:,1,:,:]*est[:,3,:,:]
        #lbl_r = lbl[:,0,:,:]*lbl[:,2,:,:] + lbl[:,1,:,:]*lbl[:,3,:,:]

        #est_i = est[:,1,:,:]*est[:,2,:,:] - est[:,0,:,:]*est[:,3,:,:]
        #lbl_i = lbl[:,1,:,:]*lbl[:,2,:,:] - lbl[:,0,:,:]*lbl[:,3,:,:]
        
        #ph_diff = torch.abs(lbl_r-est_r) + torch.abs(lbl_i-est_i)
        loss_ph_diff = torch.sum(ph_diff) / float(n_act_frames) #batch_size* n_frames * n_feats)
        #loss += 0.01*loss_ph_diff

        #playing with loss 
        #loss = loss_ri + 0.01*loss_ph_diff
        if "MIMO_RI"==self.loss_flag:
            loss = loss_ri
        elif "MIMO_RI_MAG"==self.loss_flag:
            loss = loss_ri + loss_mag
        elif "MIMO_RI_MAG_PD"==self.loss_flag:
            loss = loss_ri + loss_mag + 0.01*loss_ph_diff
        elif "MIMO_RI_PD"==self.loss_flag:
            loss = loss_ri + 0.01*loss_ph_diff
        else:
            print("Invalid loss flag")
        
        #frame level metrics
        #loss_ri_frm = torch.sum(ri,dim=3)
        #loss_mag_frm = torch.sum(mag,dim=3)
        
        # tried ILD didn't work out over IPD , code got deleted
        loss_mag_diff=0

        return loss, loss_ri, loss_mag, loss_ph_diff, loss_mag_diff

    
    def get_mag_ph(self,x):
        #est.shape : (batch, n_ch, T, F)
        (batch, n_ch, T, F) = x.shape
        est_mag_1 = torch.sqrt(x[:,0]**2 + x[:,1]**2 + self.eps)
        est_mag_2 = torch.sqrt(x[:,2]**2 + x[:,3]**2 + self.eps)

        est_ph_1 = torch.atan2(x[:,1,:,1:F-1] + self.eps, x[:,0,:,1:F-1] + self.eps)
        est_ph_2 = torch.atan2(x[:,3,:,1:F-1] + self.eps, x[:,2,:,1:F-1] + self.eps)
        """
        est_c_1 = torch.complex(est[:,0], est[:,1])
        est_c_2 = torch.complex(est[:,2], est[:,3])
        est_mag_1, est_ph_1 = torch.abs(est_c_1), torch.angle(est_c_1)
        est_mag_2, est_ph_2 = torch.abs(est_c_2), torch.angle(est_c_2)
        """
        return est_mag_1, est_ph_1, est_mag_2, est_ph_2

    def lossMask(self, labels, nframes):
        loss_mask = torch.zeros_like(labels).requires_grad_(False)
        for j, seq_len in enumerate(nframes):
            loss_mask.data[j,:,0:seq_len,:] += 1.0
        return loss_mask
    
if __name__=="__main__":
    loss = LossFunction("MISO_RI_MAG")
    #mimo_loss = MIMO_LossFunction("MIMO_RI_PD")
    x = torch.randn(6,2,50,257)
    y = torch.randn(6,2,50,257)
    _loss = loss(x,y, torch.tensor([45, 50, 35, 25, 30, 33]).squeeze())
    #_loss2 = mimo_loss(x,y,torch.tensor([45, 50, 35, 25, 30, 33]).squeeze())
    print(_loss ) #_loss2

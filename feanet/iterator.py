import string
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from functools import reduce

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    '''Random character generator'''
    return ''.join(random.choice(chars) for _ in range(size))

class PsiNet(nn.Module):
    def __init__(self, nb_layers=3, mode='thermal'):
        super(PsiNet, self).__init__()

        self.mode = mode
        self.km, self.ku, self.kf = 1, 1, 1 # thermal problem
        if(self.mode == 'elastic'):
            self.km, self.ku, self.kf = 2, 2, 2
            
        # self.attention_map = nn.Sequential(
        #     nn.Conv2d(self.km, 1, kernel_size=2, padding=1),
        #     nn.Conv2d(1, 1, kernel_size=5, padding=2),
        #     nn.Conv2d(1, self.ku, kernel_size=1, padding=0),
        #     nn.Unfold(kernel_size=3, padding=1),
        #     nn.Tanh()
        #     )

        self.smoother = nn.ModuleList([nn.Conv2d(self.ku, self.ku, 3, padding=1, bias=False)
                                         for _ in range(nb_layers)])

    def forward(self, m, x, dirich_idx):
        '''
        m: material field
        x: error between Jacobi solution and initial guess '''
        
        # bs0, ku0, h0, w0 = x.size()
        # new_x = F.unfold(x, kernel_size = 3, padding = 1).view(bs0, ku0, -1, h0, w0) # shape (bs, ku, 9, h, w)
        # attention = self.attention_map(m).view(bs0, ku0, -1, h0, w0) # shape (bs, ku, 9, h, w)
        # attention_x = new_x * attention # shape (bs, ku, 9, h, w)
        # attention_x = attention_x.sum(dim=2) # shape (bs, ku, h, w)
        return reduce(lambda acc, el: el(acc) * dirich_idx, self.smoother, x) # shape (bs, ku, h, w)
    

class PsiIterator(nn.Module):
    def __init__(self, 
                 dev,
                 h, 
                 psi_net=None,
                 grid=None,
                 n=2**5,
                 nb_layers=3,
                 batch_size=1,
                 max_epochs=1000,
                 mode='thermal',
                 iterator='jac',
                 model_dir=None):
        super(PsiIterator, self).__init__()
        self.device = dev
        self.h = h
        self.mode = mode
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.iterator = iterator
        self.loss = nn.MSELoss()
        self.grid = grid

        if(iterator == 'jac'):
            self.psi_net = None # don't use neural network model to modify Jacobi iterator
        else:
            if(psi_net == None):
                self.psi_net = PsiNet(nb_layers=nb_layers, mode=self.mode).to(self.device).double()
            else:
                self.psi_net = psi_net.to(self.device).double()
            self.optimizer = torch.optim.Adadelta(self.psi_net.parameters())
            self.model_dir=model_dir

    def PsiRelax(self, v, m, msk, d, d_idx, term_KU=None, term_F=None, h=None, f=None, t=None, t_conn=None, num_sweeps_down=1):
        '''
        Perform a fixed number of Psi iteration
        '''
        u = v.clone()
        for _ in range(num_sweeps_down):
            jac_it = self.grid.jac.jacobi_convolution(u, m, msk, d, d_idx, term_KU, term_F, h, f, t, t_conn)
            if (self.iterator == 'jac'):
                return jac_it
            else:
                return jac_it + self.psi_net(m, jac_it-u, d_idx) 

    def RandomSampling(self, x):
        u = torch.randn_like(x).double().to(self.device)
        return u

    def TrainSingleEpoch(self, train_dataloader):
        running_loss = 0.
        for i, data in enumerate(train_dataloader):
            mask_train, dirich_idx_train, dirich_value_train, traction_idx_train, traction_value_train, traction_conn_train, material_train, f_train, u_train = data
        
            #print(u_train.shape)
            self.optimizer.zero_grad() # zero the gradients for every batch
            k = 1 #random.randint(1,20)

            uu = self.RandomSampling(f_train)
            u_out = self.PsiRelax(uu, material_train, mask_train, dirich_value_train, dirich_idx_train, None, None, self.h, f_train, traction_value_train, traction_conn_train, k)
            loss_i = self.loss(u_out, u_train)
            loss_i.backward()
            self.optimizer.step()
        
            running_loss += loss_i.item()
    
        last_loss = running_loss/(i+1)
        return last_loss
    
    def Train(self, training_set):
        train_dataloader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        loss_train = torch.zeros((self.max_epochs, 1))
        avg_loss = self.TrainSingleEpoch(train_dataloader)
        loss_train[0] = avg_loss
        print('Step-0 loss:', avg_loss)

        for epoch in range(1, self.max_epochs):
            avg_loss = self.TrainSingleEpoch(train_dataloader)
            if(epoch % 50 == 0):
                print('Step-'+str(epoch)+' loss:', avg_loss)

            # save the model's state
            mpath = os.path.join(self.model_dir,self.mode+'_'+id_generator()+'.pth')
            torch.save(self.psi_net.state_dict(), mpath)
            loss_train[epoch] = avg_loss
        return loss_train
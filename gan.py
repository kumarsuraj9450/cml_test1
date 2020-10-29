from time import sleep


from tqdm.notebook import tqdm, tnrange, tqdm_notebook

import matplotlib.pyplot as plt

import numpy as np

import torch

from torchvision import datasets,transforms


from fastai.imports import *
from fastai.torch_core import *

from pathlib import Path


path = Path('/storage')
path = path/'mnist'
path.mkdir(exist_ok=True)


train = datasets.MNIST(path,
                       download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST(path,
                      train = False ,
                      download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))


trainset = torch.utils.data.DataLoader(train, batch_size=32,shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=32,shuffle=True)


def RMSELoss(yhat,y):
    return torch.sqrt(torch.nn.functional.mse_loss(yhat,y))
#     return torch.sqrt(torch.mean((yhat-y)**2))

criterion = RMSELoss

def vae_loss(orignal,reconstructed):
    pred = reconstructed.get('model_output')
    
    data_fidelity_loss = orignal * torch.log(1e-10 + pred) + (1 - orignal) * torch.log(1e-10 + 1 - pred)
    data_fidelity_loss = -torch.sum(data_fidelity_loss)
    
    mu, sigma = reconstructed.get('mu'), reconstructed.get('sigma')
    
    kl_loss = 1  + sigma - torch.pow(mu,2) - torch.exp(sigma)
    kl_loss = -0.5 * torch.sum(kl_loss)
    
    a,b = 1,1
    
    return (a*data_fidelity_loss + b*kl_loss) /2
    
criterion = vae_loss



class net(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.latent_size = 11
        self.l1 = nn.Linear(28*28,499)
        self.l2 = nn.Linear(499,359)
        self.l3 = nn.Linear(359,209)
        self.l4 = nn.Linear(209,99)
#         self.l5 = nn.Linear(99,self.latent_size)
        self.mu = nn.Linear(99,self.latent_size)
        self.sigma = nn.Linear(99,self.latent_size)
        self.l6 = nn.Linear(self.latent_size,99)
        self.l7 = nn.Linear(99,209)
        self.l8 = nn.Linear(209,359)
        self.l9 = nn.Linear(359,499)
        self.l10 = nn.Linear(499,28*28)
        
        self.fulyCnctdLayer = [self.l1, self.l2,
                               self.l3, self.l4, 
                               self.mu, self.sigma, 
                               self.l6, self.l7, 
                               self.l8, self.l9,
                              self.l10]
        
        self.drpt_layers = [nn.Dropout(.5),nn.Dropout(.5),nn.Dropout(.5),nn.Dropout(.5),nn.Dropout(.5),nn.Dropout(.5),
                     nn.Dropout(.5),nn.Dropout(.5),nn.Dropout(.5),nn.Dropout(.5)]
        
        self.bthNrm_layers = [nn.BatchNorm1d(499), nn.BatchNorm1d(359), nn.BatchNorm1d(209), nn.BatchNorm1d(99),
                                nn.BatchNorm1d(self.latent_size), nn.BatchNorm1d(self.latent_size),  
                                 nn.BatchNorm1d(99), nn.BatchNorm1d(209), nn.BatchNorm1d(359), nn.BatchNorm1d(499),]
    
        
    def basic_layer(self,x,layer_n,sigmoid=False):
        L_layer  = self.fulyCnctdLayer[layer_n]
        L_bthNrm = self.bthNrm_layers[layer_n]
        L_drpt = self.drpt_layers[layer_n]
        x = (L_bthNrm(L_drpt(L_layer(x))))
        if not sigmoid: return torch.relu(x)
        else: return torch.sigmoid(x)
               
    def forward(self,x):
        x = self.basic_layer(x,0)
        x = self.basic_layer(x,1)
        x = self.basic_layer(x,2)
        x = self.basic_layer(x,3)
        mu = self.basic_layer(x,4)
        sigma = self.basic_layer(x,5)
        epsilon = torch.empty(self.latent_size).normal_(mean=0.,std=1.)
        latent = mu + torch.exp(.5 + sigma)*epsilon
        x = self.basic_layer(latent,6)
        x = self.basic_layer(x,7)
        x = self.basic_layer(x,8)
        x = self.basic_layer(x,9)
        x = torch.sigmoid(self.l10(x))
        return {'model_output' : x ,
               'mu' : mu,
               'sigma' : sigma}
        
#     def forward(self,x):
#         x = self.l1(x).relu()
#         x = self.l2(x).relu()
#         x = self.l3(x).relu()
#         x = self.l4(x).relu()
# #         x = self.l5(x).relu()

#         mu_layer = self.mu(x).relu()
#         sigma = self.sigma(x).relu()
#         epsilon = torch.empty(11).normal_(mean=0.,std=1.)
#         latent = mu_layer + torch.exp(.5 + sigma)*epsilon
#         latent = latent

#         x = self.l6(latent).relu()
#         x = self.l7(x).relu()
#         x = self.l8(x).relu()
#         x = self.l9(x).relu()
#         x = torch.sigmoid(self.l10(x))
#         return x

    
model = net()
# print(model)
        
outfile = open('data.txt', 'w') 

opt = torch.optim.Adam(model.parameters(),lr = 1e-3, weight_decay = 3e-3 )

EPOCHS  = 10

for e in tnrange(EPOCHS, desc='EPOCHS', leave='False', unit='Epoch'):
    model.train()
    for data in tqdm(trainset, desc='Training Loop', leave='False', unit='batch'):
        x,y = data
        model.zero_grad()
        out = model(x.view(-1,28*28))
        loss = criterion(x.view(-1,28*28),out)
        loss.backward()
        opt.step()
        
    loss_batch = [] 
    
    for data in tqdm(testset, desc='Test Loop', leave='False', unit='batch'):
        model.eval()
        x,y = data
        with torch.no_grad():
            out = model(x.view(-1,28*28))
            loss = criterion(x.view(-1,28*28),out)
            loss_batch.append(loss.detach().numpy())
        
    tqdm_notebook.write(f'Epoch {e+1} loss => {np.mean([loss_batch])}')

    outfile.write(f'Epoch {e+1} loss => {np.mean([loss_batch])}')
    outfile.write("\n")

outfile.close()

torch.save(model.state_dict(),path/'model_3_epoch')
torch.save(opt.state_dict(),path/'opt_3_epoch')

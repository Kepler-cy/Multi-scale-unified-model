# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 09:56:26 2022
@author: 8
"""
import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

class Discriminator(nn.Module):  
    def __init__(self, H_Dim1, H_Dim2):
        super(Discriminator, self).__init__() 
        self.D_W1 = nn.Parameter(torch.Tensor(H_Dim1*2, H_Dim2))
        init.xavier_normal_(self.D_W1)  # xavier_normal_
        self.D_W2 = nn.Parameter(torch.Tensor(H_Dim2, H_Dim2))
        init.xavier_normal_(self.D_W2)  # xavier_normal_
        self.D_W3 = nn.Parameter(torch.Tensor(H_Dim2, 1))
        init.xavier_normal_(self.D_W3)  # xavier_normal_
        self.params=nn.ParameterList([self.D_W1, self.D_W2, self.D_W3])
        self.act_relu = nn.ReLU()  
        self.act_leaky_relu = nn.LeakyReLU()
        self.act_sig = nn.Sigmoid()
    
    def forward(self,new_x, h):
        inputs = torch.cat((new_x, h),2)  #multi graph cat: torch.Size([64, 124, 248])
        D_h1 = self.act_leaky_relu(torch.matmul(inputs, self.params[0].double()))
        D_h2 = self.act_leaky_relu(torch.matmul(D_h1, self.params[1].double()))
        D_final = self.act_sig(torch.matmul(D_h2, self.params[2].double())) #reduce dim torch.Size([64, 124, 124])
        return D_final

#%%
class Generator(nn.Module):  
    def __init__(self, H_Dim1, H_Dim2):
        super(Generator, self).__init__()  
        self.G_W1 = nn.Parameter(torch.Tensor(H_Dim1, H_Dim2))
        init.xavier_normal_(self.G_W1)  #  xavier_normal_
        self.G_W2 = nn.Parameter(torch.Tensor(H_Dim1, H_Dim2))
        init.xavier_normal_(self.G_W2)  #  xavier_no.double()rmal_
        self.G_W3 = nn.Parameter(torch.Tensor(H_Dim1, H_Dim2))
        init.xavier_normal_(self.G_W3)  #  xavier_normal_
        self.G_W4 = nn.Parameter(torch.Tensor(H_Dim1*2, H_Dim2))
        init.xavier_normal_(self.G_W4)  #  xavier_normal_
        self.G_W5 = nn.Parameter(torch.Tensor(H_Dim1, H_Dim2))
        init.xavier_normal_(self.G_W5)  #  xavier_normal_
        self.G_W6 = nn.Parameter(torch.Tensor(H_Dim1, H_Dim2))
        init.xavier_normal_(self.G_W6)  #  xavier_normal_
        self.params=nn.ParameterList([self.G_W1, self.G_W2, self.G_W3, self.G_W4, self.G_W5, self.G_W6])
        self.act_relu = nn.ReLU()  #
        self.act_leaky_relu = nn.LeakyReLU()
        self.act_tanh = nn.Tanh() #Tanh

    def forward(self, new_x, dis, pop, eco, device):
        dis_matrix = torch.tensor(dis, device=device.type).double() #124*124
        pop_matrix = torch.tensor(pop, device=device.type).double() #124*124
        
        G_h1_1 = torch.mul(dis_matrix, new_x.transpose(1,2).to(device)) #-> torch.Size([64, 124, 124]) leaky_relu
        G_h1_1 = self.act_leaky_relu(G_h1_1)
        G_h1_2 = torch.matmul(G_h1_1, self.params[0].to(device).double()) #-> torch.Size([64, 124, 124])
        G_h1_2 = self.act_leaky_relu(G_h1_2)
        
        G_h2_1 =torch.mul(pop_matrix, new_x.transpose(1,2)).double() #-> torch.Size([64, 124, 124])
        G_h2_1 = self.act_leaky_relu(G_h2_1)
        G_h2_2 = torch.matmul(G_h2_1, self.params[1].to(device).double()) #-> torch.Size([64, 124, 124])
        G_h2_2 = self.act_leaky_relu(G_h2_2)
        
        cov_results = torch.cat((G_h1_2, G_h2_2),2)  #multi graph cat: torch.Size([64, 124, 124*3])
        G_final1 = self.act_leaky_relu(torch.matmul(cov_results, self.params[3].double())) #from 3 dim to extract feature #torch.Size([64, 124, 124])
        G_final2 = self.act_leaky_relu(torch.matmul(G_final1, self.params[4].double())) #from 3 dim to extract feature #torch.Size([64, 124, 124])
        G_final = self.act_relu(torch.matmul(G_final2, self.params[5].double())) # torch.Size([64, 124, 124])
        return G_final



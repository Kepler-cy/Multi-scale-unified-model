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

class MScnn(nn.Module): 
    def __init__(self, H_Dim1, H_Dim2):
        super(MScnn, self).__init__()
        self.conv11 = nn.Sequential(
            nn.Conv1d(3, 16, 3, 1, 1, bias=False),  # [64, 1, 256]
            nn.ReLU(),
        )
        self.conv12 = nn.Sequential(
            nn.Conv1d(16, 16, 3, 1, 1, bias=False), # [64, 32, 124] #
            nn.ReLU(),
        )
        self.conv13 = nn.Sequential(
            nn.Conv1d(16, 16, 3, 1, 1, bias=False), # [128, 1, 256]
            nn.ReLU(),
        )
        self.conv14 = nn.Sequential(
            nn.Conv1d(16, 16, 3, 1, 1, bias=False), # [128, 1, 256]
            nn.ReLU(),
        )
        self.conv15 = nn.Sequential(
            nn.Conv1d(16, 1, 3, 1, 1, bias=False), # [128, 1, 256]
            nn.ReLU(),
        )
        
        self.conv21 = nn.Sequential(
            nn.Conv1d(2, 16, 3, 1, 1, bias=False),  # [64, 1, 256]
            nn.ReLU(),
        )
        self.conv22 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1, bias=False), # [64, 32, 124] 32
            nn.ReLU(),
        )
        self.conv23 = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1, bias=False), # [128, 1, 256] 32
            nn.ReLU(),
        )
        self.conv24 = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1, bias=False), # [128, 1, 256] 32
            nn.ReLU(),
        )
        self.conv25 = nn.Sequential(
            nn.Conv1d(32, 1, 3, 1, 1, bias=False), # [128, 1, 256] 32
            nn.ReLU(),
        )
        self.softmax_act = nn.Softmax()
        
    def forward(self, adj_mind, memory_county_local,memory_county_global, attraction_county, memory_community, attraction_community, device):
        batch_number, county_number, community_number = memory_county_local.shape[0], attraction_county.shape[0], attraction_community.shape[1]
        adj_mind = adj_mind.repeat(batch_number, 1, 1).float()  #torch.Size([B, 124, 80])
     
        attraction_county = attraction_county.repeat(batch_number, 1, 1).float()  #torch.Size([B, 124, 1])
        inputs_county = torch.cat((memory_county_local, memory_county_global), 2).float()
        inputs_county = torch.cat((inputs_county, attraction_county), 2).float() #torch.Size([B, 124, 3])
        inputs_county = inputs_county.transpose(1, 2).float()  #torch.Size([B, 3, 124])
        #county level
        county_out1 = self.conv11(inputs_county)
        county_out2 = self.conv12(county_out1)
        county_out3 = self.conv13(county_out2) #(B, 1, 124)
        county_out5 = self.conv15(county_out3) #(B, 1, 124)
        
        #community level
        attraction_community = attraction_community.repeat(batch_number, 1, 1).float()  #torch.Size([64, 124, 80])
        inputs_community = torch.cat((memory_community.unsqueeze(3), attraction_community.unsqueeze(3)),3).float() #torch.Size([64, 124, 80, 2])
        inputs_community = inputs_community.transpose(2, 3).float()  #torch.Size([64, 124, 2, 80])
        inputs_community = inputs_community.view(batch_number*county_number, 2, community_number) 
        community_out1 = self.conv21(inputs_community)  #(64*124, 1, 80)
        community_out2 = self.conv22(community_out1)
        community_out3 = self.conv23(community_out2) #(64*124, 1, 80)
        community_out4 = self.conv24(community_out3) #(64*124, 1, 80)
        community_out5 = self.conv25(community_out4) #(64*124, 1, 80)
        county_pro = county_out5.view(batch_number, county_number) #(64, 124) 
        county_pro_repeat = county_pro.repeat(community_number, 1, 1) #(80, 64, 124)
        county_pro_repeat_tran = county_pro_repeat.transpose(0, 1).transpose(1, 2) #(64, 124, 80)
        community_output = community_out5.view(batch_number, county_number, community_number)#(64, 124, 80)
        community_output_m_county = torch.mul(county_pro_repeat_tran, community_output)
        community_output_m_county = torch.mul(community_output_m_county, adj_mind)
        final_output = community_output_m_county.reshape(batch_number, county_number*community_number)
        return county_out5.squeeze(1), final_output, community_output,community_output_m_county,community_out5,county_out5.squeeze(1).argmax(1)
    
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:31:58 2022
Built mscnn model to achieve multi-scale human mobility prediction.
@author: 8
"""
# Packages
import os
import torch
import copy
import utils
import json
import warnings
import numpy as np
import pandas as pd
import tgan_utils
import time, operator
import matplotlib.pyplot  as plt
import torch.optim as optim
from MSCNN.MSCNN_wdh import MScnn
# from MSCNN.MSCNN_uk import MScnn
# from MSCNN.MSCNN_bsh import MScnn
from MSCNN import mscnn_utils
from torch import nn
import draw_utils

#parameter setting
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

country="USA"
# country="UK"

region = 'wdh'
# region = 'bsh'
# region = 'uk'

#data down_load
# dataset_name='Brightkite'
# dataset_name='Gowalla'
# dataset_node_number = 111
# county_number = 111
# community_number = 107

dataset_name='Weeplace'  #p_hint=0.1
# region = 'bsh'
# dataset_node_number = 124
# county_number = 124
# community_number = 80

dataset_node_number = 273
county_number = 273
community_number = 80

mb_size = 128
od_travel_pro_county, od_travel_pro_county_sum = mscnn_utils.get_tgan_results_of_county_level(dataset_name,county_number,region)
od_travel_pro_community,od_travel_pro_community_sum = mscnn_utils.get_tgan_results_of_community_level(dataset_name,county_number,community_number,region)

data_path1 = "./dataset/MSCNN_training/"+str(dataset_name)+"_training_"+str(region)+".npz"
data_path2 = "./dataset/MSCNN_training/"+str(dataset_name)+"_testing_"+str(region)+".npz"
data_path3 = "./dataset/MSCNN_training/"+str(dataset_name)+"_testing_sim_train_"+str(region)+".npz"
train_data_x_county_local,train_data_x_county_global, train_data_x_community, train_data_y_county, train_data_y_community,\
    test_data_x_county_local,test_data_x_county_global, test_data_x_community, test_data_y_county, test_data_y_community,\
        _,_,_, _,_= mscnn_utils.get_mscnn_data(data_path1,data_path2, data_path3)

#train_data_x_county = [test_len=user_num, batch,:]
# a= train_data_x_county[20][0].toarray()

# county_community_centroid={}
# with open('./dataset/USA_county_community_centroid.json', 'r',encoding='UTF-8') as f:
#     county_community_centroid = json.load(fp=f)

city_community_list_num = pd.read_csv('./dataset/city_community_list_num_'+str(region)+'.csv',index_col=0)
city_community_list_num = city_community_list_num.values

#%%construct msum model
from MSCNN.MSCNN_wdh import MScnn
mscnn = MScnn(county_number, community_number)
mscnn = mscnn.to(device)
optimizer_mscnn = optim.Adam(mscnn.parameters(), lr=0.001, weight_decay=1e-6)
mscnn.train()
criterion = nn.CrossEntropyLoss()

adj_mind = np.ones((county_number, community_number))
for i in range(county_number):
    adj_mind[i,city_community_list_num[i][0]:]=0
adj_mind = torch.tensor(adj_mind, device=device.type)

mscnn_loss1 = []
mscnn_loss2 = []
for it in range(30):
    tt=0
    county_loss_ind= 0
    community_loss_ind = 0
    for bat_num in range(len(train_data_x_county_local)):
        data_memory_county_local = mscnn_utils.data_trans_to_array1(copy.deepcopy(train_data_x_county_local[bat_num]))
        data_memory_county_global = mscnn_utils.data_trans_to_array1(copy.deepcopy(train_data_x_county_global[bat_num]))
        data_memory_community = mscnn_utils.data_trans_to_array1(copy.deepcopy(train_data_x_community[bat_num]))
        data_county_class = mscnn_utils.data_trans_to_array1(copy.deepcopy(train_data_y_county[bat_num])).reshape(len(train_data_y_county[bat_num]),-1)
        data_community_class = mscnn_utils.data_trans_to_array1(copy.deepcopy(train_data_y_community[bat_num]))
        od_attraction_county_sum = od_travel_pro_county_sum.reshape(-1,1)
        od_attraction_county_sum = od_attraction_county_sum / np.max(od_attraction_county_sum)
        data_memory_community = mscnn_utils.data_to_pro(data_memory_community)
        data_memory_county_local = mscnn_utils.data_to_pro2(data_memory_county_local)
        data_memory_county_global = mscnn_utils.data_to_pro2(data_memory_county_global)
        data_memory_county_local_tensor = torch.tensor(data_memory_county_local, device=device.type) #34*124*1
        data_memory_county_global_tensor = torch.tensor(data_memory_county_global, device=device.type) #34*124*1
        data_memory_community_tensor = torch.tensor(data_memory_community, device=device.type) #64*124*80
        data_county_class_tensor = torch.tensor(data_county_class, device=device.type) #64*124*1
        data_community_class_tensor = torch.tensor(data_community_class, device=device.type) #64*124*80
        od_attraction_county_sum_tensor = torch.tensor(od_attraction_county_sum, device=device.type) #124*1
        od_attraction_community_sum_tensor = torch.tensor(od_travel_pro_community_sum, device=device.type) #124*80
        
        optimizer_mscnn.zero_grad()
        classification_county, classification_community,aa,bb,cc,dd = mscnn(adj_mind, data_memory_county_local_tensor,data_memory_county_global_tensor, od_attraction_county_sum_tensor, \
                                      data_memory_community_tensor, od_attraction_community_sum_tensor, device)
        
        county_loss =  criterion(classification_county, data_county_class_tensor.argmax(1))
        community_loss =  criterion(classification_community, data_community_class_tensor.view(data_county_class_tensor.shape[0],-1).argmax(1))

        loss = county_loss+community_loss
        county_loss.backward(retain_graph=True)
        community_loss.backward()
        optimizer_mscnn.step()
        county_loss_ind += county_loss.detach().cpu().numpy()
        community_loss_ind += community_loss.detach().cpu().numpy()
        tt+=1
        
    mscnn_loss1.append(county_loss_ind/tt)
    mscnn_loss2.append(community_loss_ind/tt)
    if it % 1 == 0:
        print('Iter: {}'.format(it))
        print('county_loss_ind: {:.4}'.format(county_loss_ind/tt))
        print('community_loss_ind: {:.4}'.format(community_loss_ind/tt))

draw_utils.draw_loss_mscnn(mscnn_loss1[:],mscnn_loss2[:],"p")
torch.save(mscnn.state_dict(),"./model_save/mscnn0828.pth")

##%%load model
# from MSCNN.MSCNN_wdh import MScnn
# criterion = nn.CrossEntropyLoss()
# mscnn_ = MScnn(county_number, community_number)
# mscnn_ = mscnn_.to(device)
# mscnn_.load_state_dict(torch.load("./model_save/mscnn0828.pth", map_location=torch.device(device.type)))

#%% model testing
adj_mind = np.ones((county_number, community_number))
for i in range(county_number):
    adj_mind[i,city_community_list_num[i][0]:]=0
adj_mind = torch.tensor(adj_mind, device=device.type)

county_loss= 0
community_loss = 0
for bat_num in range(len(test_data_x_county_local)):
    # bat_num=0
    data_memory_county_local = mscnn_utils.data_trans_to_array1(copy.deepcopy(test_data_x_county_local[bat_num]))
    data_memory_county_global = mscnn_utils.data_trans_to_array1(copy.deepcopy(test_data_x_county_global[bat_num]))
    data_memory_community = mscnn_utils.data_trans_to_array1(copy.deepcopy(test_data_x_community[bat_num]))
    data_county_class = mscnn_utils.data_trans_to_array1(copy.deepcopy(test_data_y_county[bat_num])).reshape(len(test_data_y_county[bat_num]),-1)
    data_community_class = mscnn_utils.data_trans_to_array1(copy.deepcopy(test_data_y_community[bat_num]))
    od_attraction_county_sum = od_travel_pro_county_sum.reshape(-1,1)
    od_attraction_county_sum = od_attraction_county_sum / np.max(od_attraction_county_sum)
    data_memory_community = mscnn_utils.data_to_pro(data_memory_community)
    data_memory_county_local = mscnn_utils.data_to_pro2(data_memory_county_local)
    data_memory_county_global = mscnn_utils.data_to_pro2(data_memory_county_global)
    
    data_memory_county_local = torch.tensor(data_memory_county_local, device=device.type) #64*124*1
    data_memory_county_global = torch.tensor(data_memory_county_global, device=device.type) #64*124*1
    data_memory_community = torch.tensor(data_memory_community, device=device.type) #64*124*80
    data_county_class = torch.tensor(data_county_class, device=device.type) #64*124*1
    data_community_class = torch.tensor(data_community_class, device=device.type) #64*124*80
    od_attraction_county_sum = torch.tensor(od_attraction_county_sum, device=device.type) #124*1
    od_attraction_community_sum = torch.tensor(od_travel_pro_community_sum, device=device.type) #124*80
    
    classification_county, classification_community,aa,bb,cc,dd = mscnn_(adj_mind,data_memory_county_local,data_memory_county_global, od_attraction_county_sum, \
                                  data_memory_community, od_attraction_community_sum, device)
    
    county_loss =  criterion(classification_county, data_county_class.argmax(1))
    community_loss =  criterion(classification_community, data_community_class.view(len(test_data_y_community[bat_num]),-1).argmax(1))
    county_loss += county_loss.detach().cpu().numpy() / len(data_county_class)
    community_loss += community_loss.detach().cpu().numpy() / len(data_community_class)

print(county_loss/ len(test_data_y_county))
print(community_loss/ len(test_data_y_community))
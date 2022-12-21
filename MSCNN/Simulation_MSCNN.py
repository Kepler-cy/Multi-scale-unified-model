# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:26:39 2022
@author: 8
"""
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
import matplotlib.pyplot as plt
import torch.optim as optim
# from MSCNN.MSCNN_uk import MScnn
from MSCNN.MSCNN_wdh import MScnn
# from MSCNN.MSCNN_bsh import MScnn
from MSCNN import mscnn_utils
from torch import nn

#parameter setting
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

country="USA"
# country="UK"

region = 'wdh'
# region = 'bsh'
# region = 'uk'

#data loading
# dataset_name='Gowalla'
# dataset_node_number = 111
# county_number = 111
# community_number = 107

dataset_name='Weeplace'  #p_hint=0.1
# dataset_node_number = 124
# county_number = 124
# community_number = 80

dataset_node_number = 273
county_number = 273
community_number = 80

scale_rate = 2/3
# mb_size = 128
od_travel_pro_county, od_travel_pro_county_sum = mscnn_utils.get_tgan_results_of_county_level(dataset_name,county_number,region)
od_travel_pro_community,od_travel_pro_community_sum = mscnn_utils.get_tgan_results_of_community_level(dataset_name,county_number,community_number,region)

city_community_list_num = pd.read_csv('./dataset/city_community_list_num_'+str(region)+'.csv',index_col=0)
city_community_list_num = city_community_list_num.values.reshape(-1)

data_path1 = "./dataset/MSCNN_training/"+str(dataset_name)+"_training_"+str(region)+".npz"
data_path2 = "./dataset/MSCNN_training/"+str(dataset_name)+"_testing_"+str(region)+".npz"
data_path3 = "./dataset/MSCNN_training/"+str(dataset_name)+"_testing_sim_test_"+str(region)+".npz"
train_data_x_county_local,train_data_x_county_global, train_data_x_community, train_data_y_county, train_data_y_community,\
    test_data_x_county_local,test_data_x_county_global, test_data_x_community, test_data_y_county, test_data_y_community,\
        simulation_memory_county_local,simulation_memory_county_global,simulation_memory_community, simulation_memory_county_id,\
            simulation_memory_community_id= mscnn_utils.get_mscnn_data(data_path1,data_path2, data_path3)

user_info_boshwash, test_user = mscnn_utils.get_train_test_user1(dataset_name, county_number, community_number,region,country)

criterion = nn.CrossEntropyLoss()
mscnn_ = MScnn(county_number, community_number)
mscnn_ = mscnn_.to(device)
mscnn_.load_state_dict(torch.load("./model_save/mscnn0816.pth", map_location=torch.device(device.type)))
#Clipp matrix
adj_mind = np.ones((county_number, community_number))
for i in range(county_number):
    adj_mind[i,city_community_list_num[i]:]=0
adj_mind = torch.tensor(adj_mind, device=device.type)

od_attraction_county_sum = od_travel_pro_county_sum.reshape(-1,1)
od_attraction_county_sum = od_attraction_county_sum / np.max(od_attraction_county_sum)
od_attraction_county_sum_tensor = torch.tensor(od_attraction_county_sum, device=device.type) #124*1
od_attraction_community_sum_tensor = torch.tensor(od_travel_pro_community_sum, device=device.type) #124*80

def softmax(x, axis=1):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

county_move_info = []
community_move_info = []
for bat_num in range(len(test_data_x_county_local)):
    user_trips_list = user_info_boshwash[test_user[bat_num][0]]
    county_move_info_ind = []
    community_move_info_ind = []
    
    current_county_id = copy.deepcopy(simulation_memory_county_id[bat_num])
    current_community_id = copy.deepcopy(int(simulation_memory_community_id[bat_num]))
    data_memory_county_ind_i_local = mscnn_utils.data_trans_to_array2(copy.deepcopy(simulation_memory_county_local[bat_num])).reshape(county_number,county_number)
    data_memory_county_ind_i_global = mscnn_utils.data_trans_to_array2(copy.deepcopy(simulation_memory_county_global[bat_num])).reshape(county_number,1)
    data_memory_community_ind_i = mscnn_utils.data_trans_to_array2(copy.deepcopy(simulation_memory_community[bat_num])).reshape(county_number,community_number)
    
    for move_index in range(int(len(user_trips_list)*scale_rate), len(user_trips_list)):
        data_memory_county_temp_local = data_memory_county_ind_i_local[current_county_id,:].reshape(1, county_number, 1)
        data_memory_county_temp_global = data_memory_county_ind_i_global.reshape(1, county_number, 1) 
        data_memory_county_pro_local = mscnn_utils.data_to_pro2(data_memory_county_temp_local)
        data_memory_county_pro_global = mscnn_utils.data_to_pro2(data_memory_county_temp_global)
        data_memory_community_temp = mscnn_utils.data_to_pro(data_memory_community_ind_i.reshape(1,county_number,community_number))
        data_memory_county_tensor_local = torch.tensor(data_memory_county_pro_local, device=device.type) # 64*124*1
        data_memory_county_tensor_global = torch.tensor(data_memory_county_pro_global, device=device.type) # 64*124*1
        data_memory_community_tensor = torch.tensor(data_memory_community_temp, device=device.type) # 64*124*80
        classification_county, classification_community,\
            aa,bb,cc,dd = mscnn_(adj_mind,data_memory_county_tensor_local, data_memory_county_tensor_global, od_attraction_county_sum_tensor, \
                                                              data_memory_community_tensor, od_attraction_community_sum_tensor, device)
        county_level_pre_pro = classification_county.detach().cpu().numpy().reshape(county_number,) #1*124
        community_level_pre_pro = classification_community.detach().cpu().numpy().reshape(county_number,community_number) #(124,80)
        
        p_county = softmax(county_level_pre_pro.reshape(1,-1)).T
        #select county_number、community_number with pro
        next_county_id = np.random.choice(county_number, size=1, p=p_county.reshape(-1))[0]
        
        community_level_pre_in_county = community_level_pre_pro[next_county_id,:]
        if len(np.nonzero(community_level_pre_in_county)[0])==0:
            pp = np.ones((community_number)) #city_community_list_num
            for k in range(city_community_list_num[next_county_id],community_number):
                pp[k] = float("-inf")
            pp = softmax(pp.reshape(1,-1)).T
            next_community_id = np.random.choice(community_number, size=1, p=pp.reshape(-1))[0]
        else:
            for k in range(city_community_list_num[next_county_id],community_number):
                community_level_pre_in_county[k] = float("-inf")
            p_community = softmax(community_level_pre_in_county.reshape(1,-1)).T
            next_community_id = np.random.choice(community_number, size=1, p=p_community.reshape(-1))[0]
        #record move info
        ind_county = []
        ind_county.append(current_county_id)
        ind_county.append(next_county_id)
        county_move_info_ind.append(ind_county)
        ind_commmunity = []
        ind_commmunity.append(current_community_id)
        ind_commmunity.append(next_community_id)
        community_move_info_ind.append(ind_commmunity)
        
        #update his info
        data_memory_county_ind_i_local[current_county_id, next_county_id] += 1
        data_memory_county_ind_i_global[next_county_id] += 1
        data_memory_community_ind_i[next_county_id, next_community_id] += 1
        current_county_id = next_county_id
        current_community_id = next_community_id

    #update his info
    county_move_info.append(county_move_info_ind)
    community_move_info.append(community_move_info_ind)

#%% save data
county_move_info_dict = {}
community_move_info_dict = {}
for i in range(len(county_move_info)):
    county_move_info_dict[str(i)] = np.array(county_move_info[i], dtype=float).tolist()
    community_move_info_dict[str(i)] = np.array(community_move_info[i], dtype=float).tolist()
with open('./simulaion_results/county_move_info_dict_mscnn_'+str(region)+'.json', 'w') as f:
    json.dump(county_move_info_dict, f)
with open('./simulaion_results/community_move_info_dict_mscnn_'+str(region)+'.json', 'w') as f:
    json.dump(community_move_info_dict, f)
    
# %%
county_move_info_dict = {}
community_move_info_dict = {}
with open('./simulaion_results/county_move_info_dict_mscnn_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
    county_move_info_dict = json.load(fp=f)
with open('./simulaion_results/community_move_info_dict_mscnn_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
    community_move_info_dict = json.load(fp=f)

#%%save individual move info
county_move_info_dict_real = {}
community_move_info_dict_real = {}
# for i in range(len(train_data_x_county_local)): #2000+ ind
for i in range(len(test_data_x_county_local)): #542 ind
    # user_trips_list = user_info_boshwash[train_user[i][0]]
    user_trips_list = user_info_boshwash[test_user[i][0]] 
    county_move_info_ind = [] 
    community_move_info_ind = []
    for j in range(int(len(user_trips_list)*scale_rate),len(user_trips_list)):
        #record move info
        ind_county = []
        ind_county.append(user_trips_list[j-1][2])
        ind_county.append(user_trips_list[j][2])
        county_move_info_ind.append(ind_county)
        ind_commmunity = []
        # ind_commmunity.append(user_trips_list[j-1][3]) #*************************uk
        # ind_commmunity.append(user_trips_list[j][3])
        ind_commmunity.append(user_trips_list[j-1][4])
        ind_commmunity.append(user_trips_list[j][4])#*************************usa
        community_move_info_ind.append(ind_commmunity)
    county_move_info_dict_real[str(i)] = np.array(county_move_info_ind, dtype=float).tolist()
    community_move_info_dict_real[str(i)] = np.array(community_move_info_ind, dtype=float).tolist()

with open('./simulaion_results/county_move_info_dict_real_'+str(region)+'.json', 'w') as f:
    json.dump(county_move_info_dict_real, f)
with open('./simulaion_results/community_move_info_dict_real_'+str(region)+'.json', 'w') as f:
    json.dump(community_move_info_dict_real, f)
    

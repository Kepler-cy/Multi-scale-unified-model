# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:44:27 2022
the method of mscnn model, including TGAN model results extraction, data reconstruction, and so on.
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
import tgan_utils
import copy
import time, operator
from scipy import sparse
import matplotlib.pyplot as plt
from TGGAN.TGAN import Discriminator, Generator
import torch.optim as optim

u_id_=5
scale_rate = 2/3
min_num = 20
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tgan_results_of_county_level(dataset_name,dataset_node_number,region):
    H_Dim1, H_Dim2 = dataset_node_number, dataset_node_number
    #load model
    generator_county = Generator(H_Dim1, H_Dim2)
    generator_county = generator_county.to(device)
    generator_county.load_state_dict(torch.load("./model_save/generator_county.pth", map_location=torch.device(device.type)))
    
    data_path1 = "./dataset/TGAN_training/"+str(dataset_name)+"_"+str(region)+"/"+str(dataset_name)+"_county_"+str(region)+"2.npz"
    data_path2 = "./dataset/TGAN_training/"+str(dataset_name)+"_"+str(region)+"/"+str(dataset_name)+"_county_indicates_"+str(region)+"2.npz"
    data_raw, _, test_masking_matrix_, od_dis,od_pop_dif,od_eco_dif = tgan_utils.get_data(data_path1,data_path2)
    for i in range(len(data_raw)): 
        data_raw[i,i] = 0
    X_test = data_raw * test_masking_matrix_
    X_test = tgan_utils.data_x_to_pro_test(X_test)  
    Z_test = test_masking_matrix_ * X_test 
    Z_test = Z_test.reshape(-1,dataset_node_number,dataset_node_number)
    
    X_test = torch.tensor(X_test, device=device.type).double() #64*124*124
    test_masking_matrix_ = torch.tensor(test_masking_matrix_, device=device.type).double() #64*124*124
    Z_test = torch.tensor(Z_test, device=device.type).double() ##64*124*124
    
    g_sample = generator_county(Z_test, od_dis, od_pop_dif, od_eco_dif, device) 
    g_sample = g_sample.detach().cpu().numpy()
    g_sample = g_sample[0,:,:]
    for i in range(len(g_sample)):
        g_sample[i, i]=0
    
    data_raw_pro = tgan_utils.data_x_to_pro_test(data_raw.astype('float')) 
    g_sample_pro = tgan_utils.data_x_to_pro_test(g_sample.astype('float')) 
    test_masking_ = test_masking_matrix_.detach().cpu().numpy()

    real_nomissing = test_masking_ * data_raw_pro
    pre_missing  = (1 - test_masking_) * g_sample_pro
    od_travel_pro = real_nomissing + pre_missing
    
    for i in range(dataset_node_number):
        od_travel_pro[i][i] = 0
    
    od_travel_pro = tgan_utils.data_x_to_pro_test(od_travel_pro.astype('float'))
    od_travel_pro_sum = np.sum(od_travel_pro,axis=0)
    return od_travel_pro, od_travel_pro_sum

def get_tgan_results_of_community_level(dataset_name,dataset_county_number,dataset_community_number,region):
    H_Dim1, H_Dim2 = dataset_community_number, dataset_community_number

    data_path1 = "./dataset/TGAN_training/"+str(dataset_name)+"_"+str(region)+"/"+str(dataset_name)+"_community_"+str(region)+"1.npz"
    data_path2 = "./dataset/TGAN_training/"+str(dataset_name)+"_"+str(region)+"/"+str(dataset_name)+"_community_indicates_"+str(region)+"1.npz"
    data_raw, _, test_masking_matrix_, od_dis,od_pop_dif,od_eco_dif = tgan_utils.get_data_community(data_path1,data_path2) 
    for i in range(dataset_community_number):
        data_raw[:,i,i] = 0

    od_travel_pro_community = []
    for county_index in range(dataset_county_number):
        #respectively load model 
        generator_community = Generator(H_Dim1, H_Dim2)
        generator_community = generator_community.to(device)
        generator_community.load_state_dict(torch.load("./model_save/community_"+str(region)+"/generator_community_county"+str(county_index)+".pth", map_location=torch.device(device.type)))
        
        X_test = data_raw[county_index,:,:] * test_masking_matrix_[county_index,0,:,:] 
        X_test = tgan_utils.data_x_to_pro_test(X_test) 
        Z_test = test_masking_matrix_[county_index,0,:,:] * X_test
        Z_test = Z_test.reshape(-1,dataset_community_number,dataset_community_number)
        
        X_test = torch.tensor(X_test, device=device.type).double() #64*124*124
        test_masking_matrix_1 = torch.tensor(test_masking_matrix_[county_index,0,:,:], device=device.type).double() #64*124*124
        Z_test = torch.tensor(Z_test, device=device.type).double() ##64*124*124
        
        g_sample = generator_community(Z_test, od_dis[county_index,:,:], od_pop_dif[county_index,:,:], od_eco_dif[county_index,:,:], device)
        g_sample = g_sample.detach().cpu().numpy()
        g_sample = g_sample[0,:,:]
        for i in range(dataset_community_number):
            g_sample[i, i]=0
        
        data_raw_pro = tgan_utils.data_x_to_pro_test(data_raw[county_index,:,:].astype('float')) 
        g_sample_pro = tgan_utils.data_x_to_pro_test(g_sample.astype('float')) 
        test_masking_ = test_masking_matrix_1.detach().cpu().numpy()
        
        od_travel_pro = test_masking_ * data_raw_pro + (1 - test_masking_) * g_sample_pro
        for i in range(dataset_community_number): 
            od_travel_pro[i, i]=0
        od_travel_pro_community.append(od_travel_pro)
    
    od_travel_pro_community = tgan_utils.data_x_to_pro(np.array(od_travel_pro_community))
    od_travel_pro_community_sum = []
    for i in range(len(od_travel_pro_community)):
        od_travel_pro_community_sum.append(np.sum(od_travel_pro_community[i],axis=0))
    od_travel_pro_community_sum = np.array(od_travel_pro_community_sum)
    
    for i in range(len(od_travel_pro_community_sum)):
        max_ = np.max(od_travel_pro_community_sum[i])
        if max_ ==0:
            od_travel_pro_community_sum[i,:] = 0
        else:
            od_travel_pro_community_sum[i,:] = od_travel_pro_community_sum[i,:]/max_
    
    return od_travel_pro_community, od_travel_pro_community_sum
    
#construct the training and tesing dataset
def save_train_test_data(dataset_name, dataset_county_number, dataset_community_number,region,country): 
    user_info_boshwash={}
    with open('./dataset/'+str(dataset_name)+"_"+str(country)+'/'+str(dataset_name)+"_"+str(country)+'_user_final_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
        user_info_boshwash = json.load(fp=f)
    
    data_user_trips_num = []
    for key,value in user_info_boshwash.items():
        ind=[]
        ind.append(key)
        ind.append(len(value))
        data_user_trips_num.append(ind)
    data_user_trips_num.sort(key=operator.itemgetter(1))
    
    train_user = []
    test_user = []
    for i in range(len(data_user_trips_num)):
        if i%u_id_ == 0:
            test_user.append(data_user_trips_num[i])
        else:
            train_user.append(data_user_trips_num[i])
    
    train_data_x_county_local = []
    train_data_x_county_global = []
    train_data_x_community = []
    train_data_y_county = []
    train_data_y_community = []
    for train_index in range(len(train_user)): #len(train_user)  
        user_trips_list = user_info_boshwash[train_user[train_index][0]] 
        if len(user_trips_list)<min_num:
            continue
        train_data_x_county_ind_local = []
        train_data_x_county_ind_global = []
        train_data_x_community_ind = []
        train_data_y_county_ind = []
        train_data_y_community_ind = []
        memory_county_global = sparse.dok_matrix((dataset_county_number, 1))
        memory_county_local = sparse.dok_matrix((dataset_county_number, dataset_county_number))
        memory_community = sparse.dok_matrix(((dataset_county_number, dataset_community_number)))
        for i in range(1,int(len(user_trips_list)*scale_rate)): 
            memory_county_local[user_trips_list[i-1][2], user_trips_list[i][2]] += 1 #from county A to county B
            memory_county_global[user_trips_list[i][2]] += 1 #from county A to county B
            memory_community[user_trips_list[i][2], int(user_trips_list[i][3])] += 1
        
        for i in range(int(len(user_trips_list)*scale_rate),len(user_trips_list)):
            train_data_x_county_ind_local.append(copy.deepcopy(memory_county_local[user_trips_list[i-1][2],:].T))
            train_data_x_county_ind_global.append(copy.deepcopy(memory_county_global)) 
            train_data_x_community_ind.append(copy.deepcopy(memory_community))
            county_y = sparse.dok_matrix((dataset_county_number, 1))
            community_y = sparse.dok_matrix((dataset_county_number, dataset_community_number))
            county_y[user_trips_list[i][2]] = 1
            community_y[user_trips_list[i][2],int(user_trips_list[i][3])] = 1
            train_data_y_county_ind.append(copy.deepcopy(county_y))
            train_data_y_community_ind.append(copy.deepcopy(community_y))
            # update historical memory
            memory_county_local[user_trips_list[i-1][2], user_trips_list[i][2]] += 1
            memory_county_global[user_trips_list[i][2]] += 1
            memory_community[user_trips_list[i][2], int(user_trips_list[i][3])] += 1
            
        train_data_x_county_local.append(train_data_x_county_ind_local)
        train_data_x_county_global.append(train_data_x_county_ind_global)
        train_data_x_community.append(train_data_x_community_ind)
        train_data_y_county.append(train_data_y_county_ind)
        train_data_y_community.append(train_data_y_community_ind)
    
    train_data_x_county_local,train_data_x_county_global, train_data_x_community = np.array(train_data_x_county_local), np.array(train_data_x_county_global), np.array(train_data_x_community)
    train_data_y_county, train_data_y_community = np.array(train_data_y_county), np.array(train_data_y_community)
    
    test_data_x_county_local = []
    test_data_x_county_global = []
    test_data_x_community = []
    test_data_y_county = []
    test_data_y_community = []
    for test_index in range(len(test_user)):
        user_trips_list = user_info_boshwash[test_user[test_index][0]]
        if len(user_trips_list)<min_num:
            continue
        test_data_x_county_ind_local = []
        test_data_x_county_ind_global = []
        test_data_x_community_ind = []
        test_data_y_county_ind = []
        test_data_y_community_ind = []
        memory_county_local = sparse.dok_matrix((dataset_county_number, dataset_county_number))
        memory_county_global = sparse.dok_matrix((dataset_county_number, 1))
        memory_community = sparse.dok_matrix((dataset_county_number, dataset_community_number))
        
        for i in range(1,int(len(user_trips_list)*scale_rate)):
            memory_county_local[user_trips_list[i-1][2],user_trips_list[i][2]] += 1
            memory_county_global[user_trips_list[i-1][2]] += 1
            memory_community[user_trips_list[i][2], int(user_trips_list[i][3])] += 1
            
        for i in range(int(len(user_trips_list)*scale_rate),len(user_trips_list)):
            test_data_x_county_ind_local.append(copy.deepcopy(memory_county_local[user_trips_list[i-1][2]].T))
            test_data_x_county_ind_global.append(memory_county_global)
            test_data_x_community_ind.append(copy.deepcopy(memory_community))
            county_y = sparse.dok_matrix((dataset_county_number, 1))
            community_y = sparse.dok_matrix((dataset_county_number, dataset_community_number))
            county_y[user_trips_list[i][2]] = 1
            community_y[user_trips_list[i][2], int(user_trips_list[i][3])] = 1
            test_data_y_county_ind.append(copy.deepcopy(county_y))
            test_data_y_community_ind.append(copy.deepcopy(community_y))
            #update historical memory
            memory_county_local[user_trips_list[i-1][2],user_trips_list[i][2]] += 1
            memory_county_global[user_trips_list[i][2]] += 1
            memory_community[user_trips_list[i][2], int(user_trips_list[i][3])] += 1
        test_data_x_county_local.append(test_data_x_county_ind_local)
        test_data_x_county_global.append(test_data_x_county_ind_global)
        test_data_x_community.append(test_data_x_community_ind)
        test_data_y_county.append(test_data_y_county_ind)
        test_data_y_community.append(test_data_y_community_ind)
        
    test_data_x_county_local, test_data_x_county_global, test_data_x_community = np.array(test_data_x_county_local),np.array(test_data_x_county_global), np.array(test_data_x_community)
    test_data_y_county, test_data_y_community = np.array(test_data_y_county), np.array(test_data_y_community)
    
    np.savez("./dataset/MSCNN_training/"+str(dataset_name)+"_training_"+str(region), data_x_county_local = train_data_x_county_local, data_x_county_global = train_data_x_county_global,\
             data_x_community = train_data_x_community, data_y_county = train_data_y_county, data_y_community = train_data_y_community)
    np.savez("./dataset/MSCNN_training/"+str(dataset_name)+"_testing_"+str(region), data_x_county_local = test_data_x_county_local, data_x_county_global = test_data_x_county_global,\
             data_x_community = test_data_x_community, data_y_county = test_data_y_county, data_y_community = test_data_y_community)

def save_simulation_data_train(dataset_name, dataset_county_number, dataset_community_number,region,country):
    user_info_boshwash={}
    with open('./dataset/'+str(dataset_name)+"_"+str(country)+'/'+str(dataset_name)+"_"+str(country)+'_user_final_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
        user_info_boshwash = json.load(fp=f)
    
    data_user_trips_num = []
    for key,value in user_info_boshwash.items():
        ind=[]
        ind.append(key)
        ind.append(len(value))
        data_user_trips_num.append(ind)
    data_user_trips_num.sort(key=operator.itemgetter(1))
    
    train_user = []
    test_user = []
    for i in range(len(data_user_trips_num)):
        if i%u_id_ == 0:
            test_user.append(data_user_trips_num[i])
        else:
            train_user.append(data_user_trips_num[i])

    train_simulation_county_local = []
    train_simulation_county_global = []
    train_simulation_community = []
    train_simulation_county_id = []
    train_simulation_community_id = []
    for train_index in range(len(train_user)):
        user_trips_list = user_info_boshwash[train_user[train_index][0]]
        if len(user_trips_list)<min_num:
            continue
        memory_county_local = sparse.dok_matrix((dataset_county_number, dataset_county_number))
        memory_county_global = sparse.dok_matrix((dataset_county_number, 1))
        memory_community = sparse.dok_matrix((dataset_county_number, dataset_community_number))
        for i in range(1,int(len(user_trips_list)*scale_rate)):
            memory_county_local[user_trips_list[i-1][2],user_trips_list[i][2]] += 1
            memory_county_global[user_trips_list[i][2]] += 1
            memory_community[user_trips_list[i][2], int(user_trips_list[i][3])] += 1
        locate_county_id = user_trips_list[i-1][2]
        locate_community_id = user_trips_list[i-1][3]
        train_simulation_county_local.append(memory_county_local)
        train_simulation_county_global.append(memory_county_global)
        train_simulation_community.append(memory_community)
        train_simulation_county_id.append(locate_county_id)
        train_simulation_community_id.append(locate_community_id)

    simulation_memory_county_local, simulation_memory_county_global, simulation_memory_community = np.array(train_simulation_county_local),np.array(train_simulation_county_global), np.array(train_simulation_community)
    simulation_memory_county_id, simulation_memory_community_id = np.array(train_simulation_county_id), np.array(train_simulation_community_id)
    
    np.savez("./dataset/MSCNN_training/"+str(dataset_name)+"_testing_sim_train_"+str(region), simulation_memory_county_local = simulation_memory_county_local, simulation_memory_county_global = simulation_memory_county_global,\
             simulation_memory_community = simulation_memory_community, locate_county_id = simulation_memory_county_id, locate_community_id = simulation_memory_community_id)

def save_simulation_data_test(dataset_name, dataset_county_number, dataset_community_number,region):
    user_info_boshwash={}
    with open('./dataset/'+str(dataset_name)+"_"+str(country)+'/'+str(dataset_name)+"_"+str(country)+'_user_final_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
        user_info_boshwash = json.load(fp=f)
    
    data_user_trips_num = []
    for key,value in user_info_boshwash.items():
        ind=[]
        ind.append(key)
        ind.append(len(value))
        data_user_trips_num.append(ind)
    data_user_trips_num.sort(key=operator.itemgetter(1))
    
    train_user = []
    test_user = []
    for i in range(len(data_user_trips_num)):
        if i%u_id_ == 0:
            test_user.append(data_user_trips_num[i])
        else:
            train_user.append(data_user_trips_num[i])

    test_simulation_county_local = []
    test_simulation_county_global = []
    test_simulation_community = []
    test_simulation_county_id = []
    test_simulation_community_id = []
    for test_index in range(len(test_user)):
        user_trips_list = user_info_boshwash[test_user[test_index][0]]
        if len(user_trips_list)<min_num:
            continue
        memory_county_local = sparse.dok_matrix((dataset_county_number, dataset_county_number))
        memory_county_global = sparse.dok_matrix((dataset_county_number, 1))
        memory_community = sparse.dok_matrix((dataset_county_number, dataset_community_number))
        for i in range(1,int(len(user_trips_list)*scale_rate)):
            memory_county_local[user_trips_list[i-1][2],user_trips_list[i][2]] += 1
            memory_county_global[user_trips_list[i][2]] += 1
            memory_community[user_trips_list[i][2], int(user_trips_list[i][3])] += 1
        locate_county_id = user_trips_list[i][2]
        locate_community_id = user_trips_list[i][3]
        test_simulation_county_local.append(memory_county_local)
        test_simulation_county_global.append(memory_county_global)
        test_simulation_community.append(memory_community)
        test_simulation_county_id.append(locate_county_id)
        test_simulation_community_id.append(locate_community_id)

    simulation_memory_county_local, simulation_memory_county_global, simulation_memory_community = np.array(test_simulation_county_local), np.array(test_simulation_county_global), np.array(test_simulation_community)
    simulation_memory_county_id, simulation_memory_community_id = np.array(test_simulation_county_id), np.array(test_simulation_community_id)
    
    np.savez("./dataset/MSCNN_training/"+str(dataset_name)+"_testing_sim_test_"+str(region), simulation_memory_county_local = simulation_memory_county_local, simulation_memory_county_global = simulation_memory_county_global,\
             simulation_memory_community = simulation_memory_community, locate_county_id = simulation_memory_county_id, locate_community_id = simulation_memory_community_id)

def get_train_test_user1(dataset_name, dataset_county_number, dataset_community_number,region,country):
    user_info_boshwash={}
    with open('./dataset/'+str(dataset_name)+"_"+str(country)+'/'+str(dataset_name)+"_"+str(country)+'_user_final_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
        user_info_boshwash = json.load(fp=f)
    
    data_user_trips_num = []
    for key,value in user_info_boshwash.items():
        ind=[]
        ind.append(key)
        ind.append(len(value))
        data_user_trips_num.append(ind)
    data_user_trips_num.sort(key=operator.itemgetter(1))
    
    train_user = []
    test_user = []
    for i in range(len(data_user_trips_num)):
        if i%u_id_ == 0:
            test_user.append(data_user_trips_num[i])
        else:
            train_user.append(data_user_trips_num[i])
    
    test_user_final= []
    for test_index in range(len(test_user)):
        user_trips_list = user_info_boshwash[test_user[test_index][0]]
        if len(user_trips_list)<min_num:
            continue
        test_user_final.append(test_user[test_index])
    
    return user_info_boshwash, test_user_final

def get_train_test_user2(dataset_name, dataset_county_number, dataset_community_number,region,country):
    user_info_boshwash={}
    with open('./dataset/'+str(dataset_name)+"_"+str(country)+'/'+str(dataset_name)+"_"+str(country)+'_user_final_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
        user_info_boshwash = json.load(fp=f)
    
    data_user_trips_num = []
    for key,value in user_info_boshwash.items():
        ind=[]
        ind.append(key)
        ind.append(len(value))
        data_user_trips_num.append(ind)
    data_user_trips_num.sort(key=operator.itemgetter(1))
    
    train_user = []
    test_user = []
    for i in range(len(data_user_trips_num)):
        if i% u_id_  == 0:
            test_user.append(data_user_trips_num[i])
        else:
            train_user.append(data_user_trips_num[i])
    
    train_user_final= []
    for train_index in range(len(train_user)):
        user_trips_list = user_info_boshwash[train_user[train_index][0]] 
        if len(user_trips_list)<min_num:
            continue
        train_user_final.append(train_user[train_index])
    
    return user_info_boshwash, train_user_final

def get_mscnn_data(file_dir1, file_dir2, file_dir3):
    data1 = np.load(file_dir1,allow_pickle=True)
    train_data_x_county_local = copy.deepcopy(data1["data_x_county_local"])
    train_data_x_county_global = copy.deepcopy(data1["data_x_county_global"])
    train_data_x_community = copy.deepcopy(data1["data_x_community"])
    train_data_y_county = copy.deepcopy(data1["data_y_county"])
    train_data_y_community = copy.deepcopy(data1["data_y_community"])

    data2 = np.load(file_dir2,allow_pickle=True)
    test_data_x_county_local = copy.deepcopy(data2["data_x_county_local"])
    test_data_x_county_global = copy.deepcopy(data2["data_x_county_global"])
    test_data_x_community = copy.deepcopy(data2["data_x_community"])
    test_data_y_county = copy.deepcopy(data2["data_y_county"])
    test_data_y_community = copy.deepcopy(data2["data_y_community"])
    
    data3 = np.load(file_dir3,allow_pickle=True)
    simulation_memory_county_local = copy.deepcopy(data3["simulation_memory_county_local"])
    simulation_memory_county_global = copy.deepcopy(data3["simulation_memory_county_global"])
    simulation_memory_community = copy.deepcopy(data3["simulation_memory_community"])
    simulation_memory_county_id = copy.deepcopy(data3["locate_county_id"])
    simulation_memory_community_id = copy.deepcopy(data3["locate_community_id"])
    
    return train_data_x_county_local,train_data_x_county_global, train_data_x_community, train_data_y_county, train_data_y_community,\
        test_data_x_county_local,test_data_x_county_global, test_data_x_community, test_data_y_county, test_data_y_community,\
            simulation_memory_county_local,simulation_memory_county_global, simulation_memory_community, simulation_memory_county_id, simulation_memory_community_id

def data_trans_to_array1(arr):
    data_ = []
    for i in range(len(arr)):
        data_.append(copy.deepcopy(arr[i].toarray()))
    return np.array(data_)

def data_trans_to_array2(arr):
    data_ = []
    for i in range(arr.shape[0]):
        data_.append(copy.deepcopy(arr[i].toarray()))
    return np.array(data_)

def data_to_pro(arr):
    data = copy.deepcopy(arr)
    for i in range(len(data)):
        for j in range(len(data[i])):
            max_ = np.max(data[i,j,:])
            if max_==0:
                data[i,j, :] = 0
            else:
                data[i,j, :] = data[i,j, :]/max_
    return data

def data_to_pro2(arr):
    data = copy.deepcopy(arr)
    for i in range(len(data)):
        max_ = np.max(data[i,:,:])
        if max_==0:
            data[i,:, :] = 0
        else:
            data[i,:, :] = data[i,:, :]/max_
    return data
import argparse
import pandas as pd
import numpy as np
import os
import utils
import json
import tgan_utils
import torch, gc
import datetime
import time, operator
from torch import nn
from torch import optim
from math import sqrt, sin, cos, pi, asin
import geopandas as gpd
import copy
from scipy import sparse

#%%
def gravity_model_in_community(dataset_name,county_number,community_number,region,country):
    community_center_coor={}
    with open('./dataset/'+str(country)+'_county_community_centroid_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
        community_center_coor = json.load(fp=f)
        
    community_pop ={}
    with open("./dataset/TGAN_training/"+str(dataset_name)+"_Community_pop_all_"+str(region)+".json", 'r',encoding='UTF-8') as f:
        community_pop = json.load(fp=f)
    community_pop_ = []
    for key,va in community_pop.items():
        community_pop_.append(va)
    community_pop_ = np.array(community_pop_, dtype=int)
    community_pro = np.zeros((county_number, county_number, community_number,community_number))
    
    for i in range(county_number):
        for j in range(county_number):
            #origin
            for k1 in range(community_number):
                if str(k1) not in community_center_coor[str(i)].keys():
                    community_pro[i,j,k1,:] = 0.0
                    continue
                else:
                    lng_lat_o = community_center_coor[str(i)][str(k1)]
                #destination
                for k2 in range(community_number):
                    if str(k2) not in community_center_coor[str(j)].keys():
                        community_pro[i,j,k1,k2] = 0.0
                    else:
                        if i==j and k1==k2:
                            community_pro[i][j][k1][k2] = 0.0
                            continue
                        lng_lat_d = community_center_coor[str(j)][str(k2)]
                        dis_od = utils.earth_distance(lng_lat_o, lng_lat_d)
                        pro = (community_pop_[i][k1]*community_pop_[j][k2])* (dis_od**(-2))
                        if pro < 0:
                            pro = 0
                        community_pro[i][j][k1][k2] = pro
    
    community_pro1 = copy.deepcopy(community_pro)
    for i in range(county_number):
        for k1 in range(community_number):
            sum_ = np.sum(community_pro1[i,:,k1,:])
            if sum_<0:
                print(i)
            if sum_ ==0:
                continue
            else:
                community_pro1[i,:,k1,:] /= sum_
    
    gravity_pro_all = sparse.dok_matrix((county_number*community_number, county_number*community_number))
    for i in range(county_number):
        for j in range(county_number):
            for k1 in range(community_number):
                for k2 in range(community_number):
                    if community_pro1[i][j][k1][k2]!=0:
                        gravity_pro_all[i*community_number+k1,j*community_number+k2] = community_pro1[i][j][k1][k2]

    np.savez("./dataset/MSCNN_training/"+str(dataset_name)+"_gravity_pro_"+str(region), gravity_pro_all = gravity_pro_all)
    
#%%
def interventing_model_in_community(dataset_name,county_number,community_number,region,country):
    community_center_coor={}
    with open('./dataset/'+str(country)+'_county_community_centroid_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
        community_center_coor = json.load(fp=f)

    community_pop ={}
    community_pop = np.load("./dataset/TGAN_training/"+str(dataset_name)+"_"+str(region)+"/"+str(dataset_name)+"_Community_pop_"+str(region)+".npz",allow_pickle=True)
    community_pop_ = copy.deepcopy(community_pop["community_pop"])
    
    community_pro = np.zeros((county_number, county_number, community_number, community_number))
    for i in range(county_number):
        for k1 in range(len(community_center_coor[str(i)])):
            destination_relevance = community_pop_[i][k1]
            origin_and_distances = []
            for j in range(county_number):
                for k2 in range(len(community_center_coor[str(j)])):
                    origin_and_distances +=  [(j, k2, utils.earth_distance(community_center_coor[str(j)][str(k2)], community_center_coor[str(i)][str(k1)]))]
            origin_and_distances.sort(key=operator.itemgetter(2))
            
            sum_inside = 0.0
            for o_coun_id, o_comm_id, dis_ in origin_and_distances:
                origin_relevance = community_pop_[o_coun_id][o_comm_id]
                if (origin_relevance + sum_inside + destination_relevance)==0:
                    prob_origin_destination= 0.0
                else:
                    prob_origin_destination = destination_relevance * (1.0 / (origin_relevance + sum_inside + destination_relevance)) # 
                sum_inside += origin_relevance
                community_pro[o_coun_id, i, o_comm_id, k1] = prob_origin_destination

    community_pro1 = copy.deepcopy(community_pro)
    for i in range(county_number):
        for k1 in range(community_number):
            sum_ = np.sum(community_pro1[i,:,k1,:])
            if sum_<0:
                print(i)
            if sum_ ==0:
                continue
            else:
                community_pro1[i,:,k1,:] /= sum_
        
    del community_pro
    
    interventing_pro_all = sparse.dok_matrix((county_number*community_number, county_number*community_number))
    for i in range(county_number):
        for j in range(county_number):
            for k1 in range(community_number):
                for k2 in range(community_number):
                    if community_pro1[i][j][k1][k2]!=0:
                        interventing_pro_all[i*community_number+k1,j*community_number+k2] = community_pro1[i][j][k1][k2]
    del community_pro1

    np.savez("./dataset/MSCNN_training/"+str(dataset_name)+"_interventing_pro_"+str(region), interventing_pro_all = interventing_pro_all)
    
    
#%%
def get_gravity_pro(file_dir1):
    data1 = np.load(file_dir1,allow_pickle=True)
    gravity_pro_all = copy.deepcopy(data1["gravity_pro_all"])
    return  gravity_pro_all

def distance_model_in_community(dataset_name,county_number,community_number, region,country):
    community_center_coor={}
    with open('./dataset/'+str(country)+'_county_community_centroid_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
        community_center_coor = json.load(fp=f)
        
    community_pro = np.zeros((county_number, county_number, community_number,community_number))
    for i in range(county_number):
        for j in range(county_number):
            #origin
            for k1 in range(community_number):
                if str(k1) not in community_center_coor[str(i)].keys():
                    community_pro[i,j,k1,:] = 0.0
                    continue
                else:
                    lng_lat_o = community_center_coor[str(i)][str(k1)]
                #destination
                for k2 in range(community_number):
                    if str(k2) not in community_center_coor[str(j)].keys():
                        community_pro[i,j,k1,k2] = 0.0
                    else:
                        if i==j and k1==k2:
                            community_pro[i][j][k1][k2] = 0.0
                            continue
                        lng_lat_d = community_center_coor[str(j)][str(k2)]
                        dis_od = utils.earth_distance(lng_lat_o, lng_lat_d)
                        pro = dis_od**(-1.55)
                        if pro < 0:
                            pro = 0
                        community_pro[i][j][k1][k2] = pro
    
    community_pro1 = copy.deepcopy(community_pro)
    for i in range(county_number):
        for k1 in range(community_number):
            sum_ = np.sum(community_pro1[i,:,k1,:])
            if sum_<0:
                print(i)
            if sum_ ==0:
                continue
            else:
                community_pro1[i,:,k1,:] /= sum_
    
    del community_pro
    
    distance_pro_all={}
    for i in range(county_number):
        for k1 in range(community_number):
            distance_pro_all[str(i*community_number+k1)]={}
            for j in range(county_number):
                for k2 in range(community_number):
                    if community_pro1[i][j][k1][k2]!=0:
                        distance_pro_all[str(i*community_number+k1)][str(j*community_number+k2)]=community_pro1[i][j][k1][k2]
    del community_pro1
    with open('./dataset/MSCNN_training/'+str(dataset_name)+"_distance_pro_"+str(region), 'w') as f:
        json.dump(distance_pro_all, f)

def get_distance_pro(file_dir1):
    data1 = np.load(file_dir1, allow_pickle=True)
    distance_pro_all = copy.deepcopy(data1["distance_pro_all"])
    return distance_pro_all

def get_distance_pro2(file_dir1,dataset_name,region):
    distance_pro_all={}
    with open('./dataset/MSCNN_training/'+str(dataset_name)+"_distance_pro_"+str(region), 'r',encoding='UTF-8') as f:
        distance_pro_all = json.load(fp=f)
    return distance_pro_all

def get_interventing_pro(file_dir1):
    data1 = np.load(file_dir1, allow_pickle=True)
    distance_pro_all = copy.deepcopy(data1["interventing_pro_all"])
    return distance_pro_all
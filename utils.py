# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:02:10 2022
@author: Chenyong
"""
import math
import json
import pandas as pd
import numpy as np
import json
import datetime
import position_util
# from skmob.tessellation import tilers
import geopandas as gpd
import matplotlib.pyplot as plt
from math import sqrt, sin, cos, pi, asin
import matplotlib as mpl

from scipy.stats import pearsonr

def square_cal(area):
    sqrt_s = math.sqrt(area) #m
    bian = sqrt_s/5
    return bian

def save_flow_data(dir_path):
    data = {}
    with open('./dataset/'+str(dir_path), 'r',encoding='UTF-8') as f:
        data = json.load(fp=f)

    odflow={}
    o_flow={}
    for key, value in data.items():
        o_id = value[4]
        d_id = value[5]
        if (str(o_id),str(d_id)) not in odflow.keys():
            odflow[(str(o_id),str(d_id))]=0
            odflow[(str(o_id),str(d_id))]+=1
        else:
            odflow[(str(o_id),str(d_id))]+=1
        if str(o_id) not in o_flow.keys():
            o_flow[str(o_id)]=0
            o_flow[str(o_id)]+=1
        else:
            o_flow[str(o_id)]+=1
        
    with open('./dataset/hangzhou_between_q_o_flow_all.json', 'w') as f:
        json.dump(o_flow, f)
        
    o2d2flow = {}
    for (o, d),f in odflow.items():
        try:
            d2f = o2d2flow[o]
            d2f[d] = f
        except KeyError:
            o2d2flow[o] = {d: f}
            
    with open('./dataset/hangzhou_between_q_o2d2flow.json', 'w') as f:
        json.dump(o2d2flow, f)
            

def earth_distance(lat_lng1, lat_lng2):
    lng1, lat1  = [l*pi/180 for l in lat_lng1]
    lng2, lat2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...


def cal_dict_num(dic):
    sum_=0
    for key,value in dic.items():
        sum_+=value
    print(sum_)

def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return x

def cal_rmse(M, real_flow_missing, pre_pro, real_data):
    pre_flow_ = []
    for i in range(len(pre_pro)):
        all_pro_of_xi = pre_pro[i].flatten()
        real_flow_of_xi_all = real_data[i].flatten()
        sum_flow = np.sum(real_flow_of_xi_all)
        pre_flow_of_xi = all_pro_of_xi * sum_flow
        for j in range(len(pre_flow_of_xi)):
            pre_flow_of_xi[j] = round(pre_flow_of_xi[j])
        pre_flow_.append(pre_flow_of_xi)
    pre_flow_ = np.array(pre_flow_)
    return np.sqrt(((real_flow_missing - pre_flow_) ** 2).mean()*1.0/ np.mean(M)),\
        (abs(real_flow_missing - pre_flow_)).mean()*1.0/ np.mean(M)

def cal_cpc_value(real_flow_missing, pre_pro, real_data):
    real_all = []
    pre_all = []
    for i in range(len(pre_pro)):
        all_pro_of_xi = pre_pro[i].flatten()
        real_flow_of_xi_all = real_data[i].flatten()
        sum_flow = np.sum(real_flow_of_xi_all)
        pre_flow_of_xi = all_pro_of_xi * sum_flow
        for j in range(len(pre_flow_of_xi)):
            pre_flow_of_xi[j] = round(pre_flow_of_xi[j])
        real_all.append(real_flow_missing[i].flatten())
        pre_all.append(pre_flow_of_xi.flatten())
    real_all = np.array(real_all).reshape(-1)
    pre_all = np.array(pre_all).reshape(-1)
    if np.sum(np.minimum(real_all, pre_all))==0 or (np.sum(real_all) + np.sum(pre_all))==0:
        return 0.0
    else:
        return 2.0 * np.sum(np.minimum(real_all, pre_all)) / (np.sum(real_all) + np.sum(pre_all))

def cal_performance(M, real_flow_missing, pre_pro, real_data):
    cpc_value = cal_cpc_value(real_flow_missing, pre_pro, real_data)
    rmse_value,mae_value = cal_rmse(M, real_flow_missing, pre_pro, real_data)
    return cpc_value, rmse_value, mae_value

def cal_performance2(M, real_data, pre_):
    cpc_value = cal_cpc_value2(pre_, real_data)
    rmse_value, mae_values = cal_rmse2(M, pre_, real_data)
    return cpc_value, rmse_value,mae_values

def cal_cpc_value2(pre_flow, real_flow_missing):
    tot_fenzi= 0
    tot_fenmu= 0
    for i in range(len(pre_flow)):
        pre_sum = np.sum(pre_flow[i].flatten())
        real_sum = np.sum(real_flow_missing[i].flatten())
        if pre_sum==0:
            pre_flow_of_xi = pre_flow[i].flatten()
        else:
            pre_flow_of_xi = pre_flow[i].flatten()/pre_sum*real_sum
        for j in range(len(pre_flow_of_xi)):
            pre_flow_of_xi[j] = round(pre_flow_of_xi[j])
        real_flow_of_xi_all = real_flow_missing[i].flatten()
        tot_fenmu += (np.sum(real_flow_of_xi_all) + np.sum(pre_flow_of_xi))
        tot_fenzi += 2 * np.sum(np.minimum(real_flow_of_xi_all,pre_flow_of_xi))
    if tot_fenmu==0:
        return 0
    return tot_fenzi*1.0 / tot_fenmu

def cal_rmse2(M, pre_flow,real_flow_missing):
    pre_flow_=[]
    for i in range(len(pre_flow)):
        pre_sum = np.sum(pre_flow[i].flatten())
        real_sum = np.sum(real_flow_missing[i].flatten())
        if pre_sum==0:
            pre_flow_of_xi = pre_flow[i].flatten()
        else:
            pre_flow_of_xi = pre_flow[i].flatten()/pre_sum*real_sum
        for j in range(len(pre_flow_of_xi)):
            pre_flow_of_xi[j] = round(pre_flow_of_xi[j])
        pre_flow_.append(pre_flow_of_xi)
    pre_flow_ = np.array(pre_flow_)
    return np.sqrt(((real_flow_missing - pre_flow_) ** 2).mean()*1.0/ np.mean(M)),\
        (abs(real_flow_missing - pre_flow_)).mean()*1.0/ np.mean(M)
        
def cal_performance3(real_data, pre_):
    cpc_value = cal_cpc_value3(pre_, real_data)
    rmse_value, mae_values = cal_rmse3(pre_, real_data)
    return cpc_value, rmse_value,mae_values

def cal_cpc_value3(pre_flow, real_data):
    tot_fenzi= 0
    tot_fenmu= 0
    pre_flow_of_xi = np.array(pre_flow).flatten()
    real_flow_of_xi_all = np.array(real_data).flatten()
    tot_fenmu += (np.sum(real_flow_of_xi_all) + np.sum(pre_flow_of_xi))
    tot_fenzi += 2 * np.sum(np.minimum(real_flow_of_xi_all,pre_flow_of_xi))
    return tot_fenzi*1.0 / tot_fenmu

def cal_rmse3(pre_flow_,real_flow_missing):
    return np.sqrt(((real_flow_missing - pre_flow_) ** 2).mean()),\
        (abs(real_flow_missing - pre_flow_)).mean()
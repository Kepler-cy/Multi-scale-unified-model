
#%% Packages
import torch
import numpy as np
import utils
import torch.nn.functional as F
import copy

# Load data
def get_data(file_dir1,file_dir2): #np.array(N, T, D)
    data1 = np.load(file_dir1)
    data_raw = copy.deepcopy(data1["od_county_matrix"])
    train_masking_matrix_ = copy.deepcopy(data1["train_m_matrix"])
    test_masking_matrix_ = copy.deepcopy(data1["test_m_matrix"])
    data2 = np.load(file_dir2)
    od_county_dis = copy.deepcopy(data2["od_county_dis"])
    od_county_pop_dif = copy.deepcopy(data2["od_county_pop_dif"])
    od_county_eco_dif = copy.deepcopy(data2["od_county_eco_dif"])
    return data_raw, train_masking_matrix_, test_masking_matrix_, od_county_dis, od_county_pop_dif,od_county_eco_dif #无需管第三个参数，后续不会用到

def get_data_community(file_dir1,file_dir2):#np.array(N, T, D)
    data1 = np.load(file_dir1,allow_pickle=True)
    data_raw = copy.deepcopy(data1["od_community_matrix"])
    train_masking_matrix_ = copy.deepcopy(data1["train_m_matrix"])
    test_masking_matrix_ = copy.deepcopy(data1["test_m_matrix"])
    
    data2 = np.load(file_dir2,allow_pickle=True)
    od_community_dis = copy.deepcopy(data2["od_community_dis"])
    od_community_pop_dif = copy.deepcopy(data2["od_community_pop_dif"])
    od_community_eco_dif = copy.deepcopy(data2["od_community_eco_dif"])
    return data_raw, train_masking_matrix_, test_masking_matrix_, od_community_dis, od_community_pop_dif,od_community_eco_dif #无需管第三个参数，后续不会用到

# Load data
def get_data_pop(file_dir1):# np.array(N, T, D)
    data1 = np.load(file_dir1)
    pop = copy.deepcopy(data1["community_pop"])
    return pop

#---------------Necessary Functions------------
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)

# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

# Random sample generator for Z
def sample_Z(m, n1, n2):
    # return np.random.uniform(0., 0.01, size = [m, n1, n2])
    return np.random.uniform(0., 0.01, size = [m, n1, n2])

def sample_Label1(m, n1, n2):
    # return np.random.uniform(0., 0.01, size = [m, n1, n2])
    return np.random.uniform(0., 0.1, size = [m, n1, n2])

def sample_Label2(m, n1, n2):
    # return np.random.uniform(0., 0.01, size = [m, n1, n2])
    return np.random.uniform(-0.1, 0, size = [m, n1, n2])

# Hint Vector Generation
def sample_M(m, n1, n2, p): # mn is hint dim
    A = np.random.uniform(0., 1., size = [m, n1, n2])
    B = A > p
    C = 1.*B
    return C

def sample_rev(m, n1, n2, p):
    A = np.random.uniform(0., 1, size = [m, n1, n2])
    B = A > p
    C = 1.*B
    return C

def data_x_to_pro(data):
    datax = copy.deepcopy(data)
    for bs in range(len(datax)):
        sum_ = np.sum(datax[bs],axis = 1)
        for i in range(len(datax[bs])):
            if sum_[i]==0:
                datax[bs,i,:]=0
            else:
                datax[bs,i,:] = datax[bs,i,:] / sum_[i]
            # print(sum_.shape)
    return datax

def data_x_to_pro_test(data):
    # datax = utils.softmax(data)
    datax = copy.deepcopy(data)
    sum_ = np.sum(datax,axis = 1)
    for i in range(len(datax)):
        if sum_[i]==0:
            datax[i,:]=0
        else:
            datax[i,:] = datax[i,:] / sum_[i]
    return datax





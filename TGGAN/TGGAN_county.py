#%% Packages
import os
import torch
import copy
import utils
import draw_utils
import json
import warnings
import numpy as np 
import tgan_utils
import matplotlib.pyplot  as plt
import torch.optim as optim
from TGGAN.TGAN import Discriminator, Generator

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# country='USA'
country='UK'

dataset_name='Gowalla'
# dataset_name='Brightkite'
region = 'uk'
dataset_node_number = 111

# dataset_name='Weeplace'  #p_hint=0.1
# region = 'bsh'
# region = 'wdh'
# dataset_node_number = 273
# dataset_node_number = 124

mb_size = 256
p_hint = 0.15 #Hint rate
alpha1 = 1.5 #Loss2 Hyperparameters
alpha2 = 1.5 #Loss3 Hyperparameters
train_len = 2000
#Weeplace_bsh Gowalla_uk  Weeplace_wdh
data_path1 = "./dataset/TGAN_training/Gowalla_uk/"+str(dataset_name)+"_county_"+str(region)+"1.npz"
data_path2 = "./dataset/TGAN_training/Gowalla_uk/"+str(dataset_name)+"_county_indicates_"+str(region)+"1.npz" 
data_raw, train_masking_matrix_, test_masking_matrix_, od_dis,od_pop_dif,od_eco_dif = tgan_utils.get_data(data_path1,data_path2) #L*Node
Dim, H_Dim1, H_Dim2 = dataset_node_number, dataset_node_number, dataset_node_number

def discriminator_loss(X, M, T, Z, H, H1, DIS, POP, ECO, Device):
    G_sample = generator_(Z, DIS, POP, ECO, Device) # Generator rsults: torch.Size([64, 124, 124]) 
    Hat_New_X = X * M*T + G_sample * (1-M*T) # Combine with original data 
    D_prob = discriminator_(Hat_New_X, H) # Discriminator results: torch.Size([64, 124, 124])
    #Loss
    D_loss = -torch.mean((1-H1) * M*T * torch.log(D_prob + 1e-8) + (1-H1) * (1-M*T) * torch.log(1. - D_prob + 1e-8)) / torch.mean(1-H1)
    return D_loss, D_prob

def generator_loss(X, X_P, M, T, Z, H, H1,  DIS, POP, ECO, Device):
    G_sample = generator_(X, DIS, POP, ECO, Device) # Generator results
    Hat_New_X = X * M*T + G_sample * (1-M)*T # Combine with original data
    # Generator Loss1
    D_prob = discriminator_(Hat_New_X, H)  # Discriminator results
    G_loss1 = - torch.mean((1-H1) * (1-M*T) * torch.log(D_prob + 1e-8)) / torch.mean((1-H1) * (1-M*T))
    # G_loss1 = torch.mean((1-H1) * (1-M*T) * torch.log((1-D_prob) + 1e-8)) / torch.mean((1-H1) * (1-M*T))

    # Generator Loss2
    G_loss2 = (torch.mean((M*T*X - M*T*G_sample)**2) / torch.mean(M*T))
    #total loss
    G_loss = (G_loss1 + alpha1 * torch.sqrt(G_loss2)) # Generator Loss_fina    + alpha2 * torch.sqrt(G_loss3)
    return G_loss, G_loss1, G_loss2#, G_loss3

#%% Start Iterations
data_raw, train_masking_matrix_, test_masking_matrix_, od_dis,od_pop_dif,od_eco_dif = tgan_utils.get_data(data_path1,data_path2) #L*Node
discriminator_ = Discriminator(H_Dim1, H_Dim2)
discriminator_ = discriminator_.to(device)
generator_ = Generator(H_Dim1, H_Dim2)
generator_ = generator_.to(device)

optimizer_D = optim.Adam(discriminator_.parameters(), lr=0.001, weight_decay=1e-6)
optimizer_G = optim.Adam(generator_.parameters(), lr=0.001, weight_decay=1e-6)
discriminator_.train() 
generator_.train()

d_loss, g_loss = [], []
for it in range(1000):
    mb_idx = tgan_utils.sample_idx(train_len, mb_size)
    M_mb = train_masking_matrix_[mb_idx, :]
    X_part = tgan_utils.data_x_to_pro_test(data_raw * test_masking_matrix_) 
    X_mb = tgan_utils.data_x_to_pro(data_raw * M_mb * test_masking_matrix_)  
    # Z_mb = tgan_utils.sample_Z(mb_size, Dim, Dim) #random matrix
    Z_mb = M_mb * test_masking_matrix_ * X_mb #+ (1-M_mb * test_masking_matrix_) * Z_mb
    H_mb1 = tgan_utils.sample_M(mb_size, Dim, Dim, p_hint) #hint_sample
    H_mb = M_mb * test_masking_matrix_ * H_mb1 + 0.5 * (1 - H_mb1)    #hint matrix
    
    Z_mb = torch.tensor(Z_mb, device=device.type).float() ##64*124*124
    X_mb = torch.tensor(X_mb, device=device.type).float() #64*124*124
    X_part = torch.tensor(X_part, device=device.type).float() #64*124*124
    M_mb = torch.tensor(M_mb, device=device.type).float() #64*124*124
    M_test = torch.tensor(test_masking_matrix_, device=device.type).float() #64*124*124
    H_mb1 = torch.tensor(H_mb1, device=device.type).float() #64*124*124
    H_mb = torch.tensor(H_mb, device=device.type).float() #64*124*124
    
    optimizer_D.zero_grad()
    D_loss_curr,d_pro = discriminator_loss(X = X_mb, M=M_mb, T=M_test, Z=Z_mb, H=H_mb, H1=H_mb1, \
                                           DIS = od_dis, POP = od_pop_dif, ECO = od_eco_dif, Device = device)
    D_loss_curr.backward()
    optimizer_D.step()
    d_loss.append(D_loss_curr.detach().cpu().numpy())
    
    optimizer_G.zero_grad()
    G_loss_curr, dloss, MSE_train_loss_curr= generator_loss(X=X_mb, X_P = X_part, M=M_mb, T=M_test, Z=Z_mb, H=H_mb,H1=H_mb1,\
                                                                             DIS = od_dis, POP = od_pop_dif, ECO = od_eco_dif, Device = device)
    G_loss_curr.backward()
    optimizer_G.step()
    g_loss.append(G_loss_curr.detach().cpu().numpy())
    
    # # Intermediate Losses
    if it % 200 == 0:
        print('Iter: {}'.format(it))
        print('D: {:.4}'.format(D_loss_curr))
        print('Train_loss1: {:.4}'.format(dloss))
        print('Train_loss2: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
        # print('Train_loss3: {:.4}'.format(np.sqrt(imp_loss.item())))

d_loss = np.array(d_loss)
g_loss = np.array(g_loss)
draw_utils.draw_loss(d_loss[:],g_loss[:],"p")
draw_utils.draw_loss(d_loss[100:],g_loss[100:],"p")

torch.save(discriminator_.state_dict(),"./model_save/discriminator_county_0823.pth")
torch.save(generator_.state_dict(),"./model_save/generator_county_0823.pth")

#%% 
discriminator_ = Discriminator(H_Dim1, H_Dim2)
discriminator_ = discriminator_.to(device)
generator_ = Generator(H_Dim1, H_Dim2)
generator_ = generator_.to(device)
discriminator_.load_state_dict(torch.load("./model_save/discriminator_county_0822.pth", map_location=torch.device(device.type)))
generator_.load_state_dict(torch.load("./model_save/generator_county_0822.pth", map_location=torch.device(device.type)))

#%%  
def test_loss(X, M, Z, DIS, POP, ECO, Device): #test loss 
    G_sample = generator_(Z, DIS, POP, ECO, Device) 
    MSE_test_loss1 = (torch.mean((M*X - M*G_sample)**2) / torch.mean(M)) # MSE Performance metric
    return MSE_test_loss1, G_sample 

# data_raw, train_masking_matrix_, test_masking_matrix_, od_dis,od_pop_dif,od_eco_dif = tgan_utils.get_data(data_path1,data_path2) #L*Node
mb_size = 1
for i in range(len(data_raw)): 
    data_raw[i, i] = 0

X_test = data_raw * test_masking_matrix_ # 20%
X_test = tgan_utils.data_x_to_pro_test(X_test)
# Z_test = tgan_utils.sample_Z(mb_size, Dim, Dim) #random matrix
Z_test = test_masking_matrix_ * X_test #+ (1-test_masking_matrix_) * Z_test
Z_test = Z_test.reshape(-1,dataset_node_number,dataset_node_number)

X_test = torch.tensor(X_test, device=device.type).float() #64*124*124
test_masking_matrix_ = torch.tensor(test_masking_matrix_, device=device.type).float() #64*124*124
Z_test = torch.tensor(Z_test, device=device.type).float() ##64*124*124

D_recon_x_loss, g_sample = test_loss(X=X_test, M=test_masking_matrix_, Z=Z_test, \
                                 DIS = od_dis, POP = od_pop_dif, ECO = od_eco_dif, Device = device)
    
g_sample = g_sample.detach().cpu().numpy()
g_sample = g_sample[0,:,:]
for i in range(len(g_sample)):
    g_sample[i, i] = 0

data_raw_pro = tgan_utils.data_x_to_pro_test(data_raw.astype('float')) 
g_sample_pro = tgan_utils.data_x_to_pro_test(g_sample.astype('float')) 
test_masking_ = test_masking_matrix_.detach().cpu().numpy()

for i in range(dataset_node_number):
    test_masking_[i][i] = 1

real_nomissing = test_masking_ * data_raw_pro
real_missing = (1 - test_masking_) * data_raw_pro
pre_nomissing  = test_masking_ * g_sample_pro
pre_missing  = (1 - test_masking_) * g_sample_pro

cpc,rmse,mae = utils.cal_performance(1-test_masking_, (1-test_masking_)*data_raw, pre_missing, data_raw) 
print(cpc)
print(rmse)
print(mae)

# %%vis
ori_num= 1
des_num= 30000
if region=="wdh":
    des_num = 2000
bin_num = 15

county_center_coor={}
with open('./dataset/USA_county_centroid_'+str(region)+'.json', 'r',encoding='UTF-8') as f:
    county_center_coor = json.load(fp=f)

from Visualization import Visualization_prediction_distance_function
fre_real,fre_pre = Visualization_prediction_distance_function.save_distance_distribution(dataset_name, "GGAN model", bin_num, 1-test_masking_, (1-test_masking_)*data_raw, pre_missing, data_raw,county_center_coor)

# from Visualization import Visualization_prediction_map_line_in_taz_function
# Visualization_prediction_map_line_in_taz_function.draw(dataset_name, "TGAN model", 1-test_masking_, (1-test_masking_)*data_raw, pre_missing, data_raw, county_center_coor)

# from Visualization import Visualization_prediction_scatter_bin_function
# df = Visualization_prediction_scatter_bin_function.draw_scatter_bin(dataset_name, "GGAN model", bin_num, 1-test_masking_, (1-test_masking_)*data_raw, pre_missing, data_raw, ori_num,des_num)


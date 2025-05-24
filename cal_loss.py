import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2

target_var='fws'
# Load data
#gnn_preds = np.load(r'./outputs_single/t2m_10_outputs_20.npy')
#gnn_preds = np.load(r'./outputs_ori/t2m_30_outputs_1.npy')
gt = np.load(r'./enspreds/gt_'+target_var+'.npy')
ec_preds = np.load(r'./enspreds/ec_'+target_var+'_ens.npy')
cfs_preds = np.load(r'./enspreds/cfs_'+target_var+'_ens.npy')
all_preds = np.load(r'./enspreds/all_'+target_var+'_ens.npy')

# Cal RMSE
rmses_pointly=np.zeros([4,91,21])
for i in range(21):
    gnn_preds = np.load(r'./outputs/'+target_var+'_'+str(i+10)+'_outputs.npy')
    for j in range(91):
        #print(r2(gt, gnn_preds),r2(gt, ec_preds[:,i,:]),rmse(gt, cfs_preds[:,i,:]),rmse(gt, all_preds[:,i,:]))
        rmses_pointly[0,j,i]=rmse(gt[:,j], gnn_preds[:,j])
        rmses_pointly[1,j,i]=rmse(gt[:,j], ec_preds[:,i,j])
        rmses_pointly[2,j,i]=rmse(gt[:,j], cfs_preds[:,i,j])
        rmses_pointly[3,j,i]=rmse(gt[:,j], all_preds[:,i,j])
np.save('./rmses_pointly_fws.npy',rmses_pointly)

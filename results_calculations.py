# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:05:30 2021

@author: caojuan
"""
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import glob
import os

print(os.getcwd())

os.chdir('E:\\SCI\\SCI6')
path='E:\\SCI\\SCI6\\2021data\\rawdata\\GEE\\soybean\\results'
files= os.listdir(path)

outpath='E:\\SCI\\SCI6\\2021data\\processdata\\GEE\\soybean\\results_analysis'


# In[1] main function
from scipy import stats
def rsquared(x, y): 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    return r_value

def accuracy(ori_data,pre_data):
    evalation_indices=pd.DataFrame(columns=['test_mae','test_rmse','test_r2','test_r2_adjust'])
    evalation_indices.loc[0,'test_mae']=mean_absolute_error(ori_data,pre_data)
    evalation_indices.loc[0,'test_rmse']=np.sqrt(mean_squared_error(ori_data,pre_data))
    evalation_indices.loc[0,'test_r2']=r2_score(ori_data,pre_data)
    evalation_indices.loc[0,'test_r_adjust']=rsquared(ori_data,pre_data)
    return evalation_indices


def index_ID(data):
    evalation_indices=pd.DataFrame(columns=['ID','test_mae','test_rmse','test_r2','test_r_adjust'])
    for ID in data['ID'].unique():
        data_one=data[data['ID']==ID]
        evalation_indice=accuracy(data_one['lables_raw'],data_one['predic_raw'])
        evalation_indice['ID']=ID
        evalation_indices=evalation_indices.append(evalation_indice,ignore_index=True)     
        print(evalation_indices)
    return evalation_indices

# In[1] main function
packages=['rf','xgb','LASSO']
lags=[0]

most_importances=pd.DataFrame(columns=['ID','column','importance','model','lag','NAME_0'])
results_importances=pd.DataFrame(columns=['ID','column','importance','model','lag','NAME_0'])
evalation_indices=pd.DataFrame(columns=['ID','test_rmse','test_r2','test_r_adjust','model','lag','NAME_0'])
datas=pd.DataFrame(columns=['ID','year','lables_raw','predic_raw','model','lag','NAME_0'])
 
for file in files:
  for package in packages:
        for lag in lags:
            path2 = path+'\\'+file+'\\'+package+'\\'+str(lag)+'\\'+str(3)
            files_results=glob.glob(path2+'\\'+"*.csv")
            results_data=pd.read_csv(files_results[1])
            # results_importance=pd.read_csv(files_results[0])
            evalation_indice=index_ID(results_data)##
            
            evalation_indice['model']=package
            evalation_indice['lag']=lag
            evalation_indice['NAME_0']=file
            
            results_data['model']=package
            results_data['lag']=lag
            results_data['NAME_0']=file  
            
            # results_importance=results_importance.groupby(['column','ID'])['importance'].mean().reset_index().sort_values(by=['ID','importance'],ascending=(True,False))
            # results_importance['model']=package
            # results_importance['lag']=lag
            # results_importance['NAME_0']=file     
            
            # most_importance=results_importance.drop_duplicates(subset='ID', keep='first')
            # most_importance['model']=package
            # most_importance['lag']=lag
            # most_importance['NAME_0']=file 
            
            # most_importances=most_importances.append(most_importance,ignore_index=True)
            datas=datas.append(results_data,ignore_index=True)
            # results_importances=results_importances.append(results_importance,ignore_index=True)
            evalation_indices=evalation_indices.append(evalation_indice,ignore_index=True)
          ##挑选出不能建立模型的行政单元
          
# evalation_indices.to_csv(outpath+'maize_all_index.csv',index=False)
# results_importances.to_csv(outpath+'maize_all_results_importances.csv',index=False)
# most_importances.to_csv(outpath+'maize_all_most_importances.csv',index=False)

# =============================================================================
# 如果test_r2小于0，表示null model

evalation_indices['test_r2_adjust']=evalation_indices['test_r_adjust']*evalation_indices['test_r_adjust']
data11=evalation_indices[evalation_indices['test_r2']>=0]
data12=evalation_indices[evalation_indices['test_r2']<0]
data12['test_r2_adjust']=0
datas1=datas.copy()
datas1['NAME_0']='Global'
datas=datas.append(datas1,ignore_index=True)
datas=datas.replace('rf','RF').replace('xgb','XGB')


evalation_indices=data12.append(data11,ignore_index=True)
datacopy=evalation_indices.copy()
datacopy['NAME_0']='Global'
evalation_indices=evalation_indices.append(datacopy,ignore_index=True)
evalation_indices=evalation_indices.replace('rf','RF').replace('xgb','XGB')
###选择每个主要的国家
evalation_indices=evalation_indices[evalation_indices.loc[:,'test_r2_adjust']!=0]

##namelist=['United States','China','Brazil','Argentina','Mexico','Global'] #maize
##namelist=['China','India','Russia','United States','France','Global'] # wheat
##namelist=['China','India','Indonesia','Bangladesh','Vietnam','Global'] # rice

namelist=['United States','Brazil','Argentina','China','Paraguay','Global'] #soybean

datas['NAME_model']=datas['NAME_0']+'_'+datas['model']

evalation_indices=evalation_indices[evalation_indices['NAME_0'].apply(lambda x: x in namelist)]
datas=datas[datas['NAME_0'].apply(lambda x: x in namelist)]
datas['predic_raw'],datas['lables_raw']=datas['predic_raw']/1000,datas['lables_raw']/1000
evalation_indices.to_csv(outpath+'//soybean_all_index.csv',index=False)

datas.to_csv(outpath+'//soybean_alldata.csv',index=False)
# =============================================================================


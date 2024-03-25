# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:42:33 2021

@author: caojuan
"""

import numpy as np
import pandas as pd
import glob
import os
path='E:\\SCI\\SCI6\\2021data\\processdata\\GEE\\Maize\\table\\'
files= os.listdir(path)
# In[1] 数据分组
Mean = ['tmean','pr','soil','vap','vpd']
Extreme= ['EA','IOD','NAO','Nino34','PDO','SAM','TSA']
local_manage=['elevation','latitude','longitude','T_CLAY','T_GRAVEL','T_OC','T_PH_H2O','T_REF_BULK','T_SAND','T_SILT']
Tech=['Irrigation','Fertilizer']

allfeature=Mean+Extreme+local_manage+Tech

##添加字符串对于一个list
def add_str(gs,s,flag):
    if flag==1:
        for i in range(len(gs)):  
            gs[i] = s + str(gs[i])
    else:
        for i in range(len(gs)):  
            gs[i] = str(gs[i])+s
    
    return tuple(gs.copy())

##求取生育期的均值
def gs_mean(data,sequential):
    gs_mean=pd.DataFrame(columns=sequential)
   # gs_mean=pd.DataFrame(columns=add_str(feature2,'_mean',0))
    for feature in sequential:
        mean_feature=data[data.columns[data.columns.str.startswith(feature)]].mean(axis=1)
        gs_mean[feature]=mean_feature
    return gs_mean
##选取不同组合

##输出每个国家的原始数据

##flag=0，只是mean climate
##flag=1，极端气候数据，
##flag=2，极端+mean climate 
##flag=3 mean + 极端+ local_manage+Tech.  ##有了这个是不是就可以不去趋势了


def select_combined(data,flag):
    if flag==0:
        feature=Mean+['Yield']+['ID']
    if flag==1:
        feature=Extreme+['Yield']+['ID']
    if flag==2:
        feature=Mean+Extreme+['Yield']+['ID']
    if flag==3:
        feature=Mean+Extreme+local_manage+Tech+['Yield']+['ID']
    return feature
    
        



##滞后月份选取数据
def selectdata(data,lag,plant,harvest,dta_comnied,mean):
    feature=select_combined(data,dta_comnied)##
    data=data[data.columns[data.columns.str.startswith(tuple(feature))]]
    if lag==0:
        data=data
    elif lag==1:
        if plant<harvest:
            gs= list(range(int(plant),int(harvest),1))
            gs=add_str(gs,'_',1)
            data=data[data.columns[data.columns.str.endswith(gs)]]
        else:
            if harvest==1:
                gs=list(range(int(plant),int(13),1))
                data=data[data.columns[data.columns.str.endswith(gs)]]
            else:
                gs1=list(range(int(plant),int(13),1))
                gs2=list(range(1,int(harvest),1))
                gs=gs1+gs2
                data=data[data.columns[data.columns.str.endswith(gs)]]
    else:
        if plant<harvest:
            gs= list(range(int(plant),int(harvest-1),1))
            gs=add_str(gs,'_',1)
            data=data[data.columns[data.columns.str.endswith(gs)]]
        else:
            if harvest==1:
                gs=list(range(int(plant),int(12),1))
                data=data[data.columns[data.columns.str.endswith(gs)]]
            elif harvest==2:
                gs=list(range(int(plant),int(13),1))
                data=data[data.columns[data.columns.str.endswith(gs)]]
            else:
                gs1=list(range(int(plant),int(13),1))
                gs2=list(range(1,int(harvest-1),1))
                gs=gs1+gs2
                data=data[data.columns[data.columns.str.endswith(gs)]]   
    if mean==True:
        return gs_mean(data,feature)
    else:
        
        return data ##gs_mean(data,feature) ##看需不需要取均值
###去趋势去趋势去趋势


# In[1] main function


# =============================================================================
# lag=0##设置是否需要提前，但是注意最多只能提前两个月，因为有些地方只有三个月生长季
# package ’rf‘,'lgbm','xgb','LASSO' rf调参较慢
# num_evals 调参次数
# mean True :取生育期均值，False,不取生育期均值
#dta_comnied 0,1,2,3 参见select_combined函数
# =============================================================================
###一定要预设上诉参数

for file in files:
    path2 = path+'\\'+file
    for file1 in glob.glob(path2+'\\'+"*.csv"):
        data=pd.read_csv(file1,na_values=0)
        plant=data['plant'].unique()
        harvest=data['harvest'].unique()
        ##data=data.fillna(method='ffill',axis=0) 前面已经插补过了
        ###挑选数据##所有全部都是树模型，所以不用标准化
        data=selectdata(data,lag,plant,harvest,dta_comnied,mean) 
        evalation_indices,labels_two_RSP=result_RSP(data,package,num_evals)
        path_output='E:\\SCI\\SCI6\\2021data\\processdata\\GEE\\Maize\\results\\'+file+'\\'+package+'\\'+str(lag)+'\\'+str(dta_comnied)+'\\'
        mkdir(path_output)
        evalation_indices.to_csv(path_output+str(int(plant))+'_'+str(int(harvest))+'_results_accurary.csv',index=False)
        labels_two_RSP.to_csv(path_output+str(int(plant))+'_'+str(int(harvest))+'_predict_results.csv',index=False)
        ##运行模型保存结果
        
        
        

###数据标准化
###调用模型_优化模型_保存模型
        print(1)

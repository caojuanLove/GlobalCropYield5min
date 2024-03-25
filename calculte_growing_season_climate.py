# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 22:07:04 2021

@author: caojuan
"""
# In[1]
import numpy as np
import pandas as pd
import os

os.chdir(r'E:\SCI\SCI6\2021data\rawdata\GEE\maize')
data=pd.read_csv('MaizeFinal1981_2015.csv',na_values=0)
CMI=pd.read_csv('CMI.csv',na_values=0)
static_data=pd.read_csv('static_Maize.csv',na_values=0)
#data['tmean_1']=data['tmmx_1']+data['tmmx_1']
#In[1] define function
def tmean(data):
    for i in list(range(1,13)):
      data['tmean_'+str(i)]=data['tmmx_'+str(i)]+data['tmmx_'+str(i)]  
    return data 
      
def add_str(gs):
    s='_'
    for i in range(len(gs)):  
        gs[i] = s + str(gs[i])
    return tuple(gs)
##删选plant>harvest的数据

def select_data1(data,gs):
    data=data[data['year']<2015]##1981-2014
    data=data.sort_values(by='year')
    data=data[data.columns[data.columns.str.endswith(gs)]]
    return data

def select_data2(data,gs):
    data=data[data['year']>1981]##1981-2014
    data=data[data.columns[data.columns.str.endswith(gs)]]
    return data

##创建目录
def mkdir(path):

    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)

    if not isExists:
        os.makedirs(path) 
        print (path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False
 

# In[链接静态指标] 
data=tmean(data)
data=data.merge(CMI,on='year')
# In[1] 
feature_static=['T_CLAY','T_GRAVEL','T_OC','T_PH_H2O','T_REF_BULK','T_SAND','T_SILT','longitude','latitude','elevation']

data_static = data[feature_static]
#data['harvest']=data['harvet']
data_static['ID'],data_static['year']=data['ID'],data['year']
#data_static=data_static.drop_duplicates(subset=['ID'], keep='first', inplace=False)
data_dynamic = data.drop(feature_static,axis=1)
data_dynamic =data_dynamic.drop(['Uniquename','NAME_1','NAME_2','NAME_3'],axis=1)


df=data_dynamic.groupby(['plant','harvest','NAME_0'])
##这里取值还是不对。1981年的并没有被取
for i in df:
    plant=int(i[0][0])
    harvest=int(i[0][1])
    Name=i[0][2]
    data_group=i[1]
    ## data_group=data_dynamic[data_dynamic['NAME_0']=='Argentina']
    ## example=data_group[data_group['ID']==1736]
    if plant>harvest:##如果plant大于harvest日期
        gs1= list(range(plant,13,1))
        gs1=add_str(gs1)
        data1=select_data1(data_group,gs1)
        gs2= list(range(1,harvest+1,1))
        gs2=add_str(gs2)
        data2=select_data2(data_group,gs2)##记得删除名字字
        data_group=data_group[data_group['year']>1981]
        data_gs=pd.concat([data1.reset_index(),data2.reset_index()],axis=1).drop(['index'],axis=1)
        data_gs=pd.concat([data_group[['ID','NAME_0','plant','harvest','year','Yield']].reset_index(),data_gs.reset_index()],axis=1).drop(['index'],axis=1)  
        del gs1,data1,gs2,data2,data_group
    else:
        gs= list(range(int(plant),harvest+1,1))
        gs=add_str(gs)
        data1=select_data2(data_group,gs)##记得删除名字字
        data_group=data_group[data_group['year']>1981]
        data_gs=pd.concat([data_group[['ID','NAME_0','plant','harvest','year','Yield']],data1],axis=1)
        
        del gs,data1
    path='E:\\SCI\\SCI6\\2021data\\rawdata\\GEE\\maize\\tableori\\'+Name+'\\'
    mkdir(path)
    data_gs=data_gs.merge(static_data,on='ID').merge(data_static,on=['ID','year'])
    ##填补空缺值，主要是分为一种：按照年groupby ，相邻单元格插补，另外一种某一年都为0 ，所以只有用前后均值补充
    data_gs=data_gs.groupby('year').apply(lambda x: x.ffill().bfill())
    data_gs=data_gs.fillna(data_gs.mean())
    data_gs.to_csv(path+Name+'_'+str(plant)+'_'+str(harvest)+'.csv',index=False,na_rep=0)
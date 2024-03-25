# -*- coding: utf-8 -*-
"""
Created on Thu May 13 23:33:11 2021

@author: Administrator
"""

import numpy as np
import pandas as pd
import glob

path='E:\\SCI\\SCI6\\data\\climate\\txt'

data=pd.DataFrame(columns={'year','month','station','tmin','tmax','tmean','pre'})

for file1 in glob.glob(path+'\\'+"*.txt"):
    data1=pd.read_csv(file1,sep=' ',index_col=False)
    data1=data1[['V04001','V04002','V01000','V12212','V12211','V11042','V13011']]
    data1.columns=['year','month','station','tmin','tmax','tmean','pre']
    data=data.append(data1,ignore_index=True)
    
data[data==32766]=np.nan

data['pre'],data['tmin'],data['tmax'],data['tmean']=data['pre']/10,data['tmin']/10,data['tmax']/10,data['tmean']/10


data=data[(data['year']<2017)&(data['year']>1980)]


data2=pd.read_csv(r'E:\SCI\SCI6\data\climate\GEE.csv',index_col=False)
data2['year'],data2['month']=data2['date'].str.split('/', expand=True)[0],data2['date'].str.split('/', expand=True)[1]


data2['GEE_tmean']=(data2['GEE_tmmn']+data2['GEE_tmmx'])/2
data2['GEE_tmmn'],data2['GEE_tmean'],data2['GEE_tmmx']=data2['GEE_tmmn']/10,data2['GEE_tmean']/10,data2['GEE_tmmx']/10


data2=data2.drop(['date'],axis=1).astype('float64')
data=data.astype('float64')

alldata=pd.merge(data2,data,how='left',on=['station','year','month'])


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from plotnine import * 

# Generate fake data


# Calculate the point density


alldata=alldata.dropna(how='any')
alldata_pre=alldata[(alldata['pre']<500)&(alldata['GEE_pr']<500)]

alldata[['GEE_pr','GEE_tmmx','GEE_tmmn','GEE_tmean','tmax','pre','tmin','tmean']] = StandardScaler().fit_transform(alldata[['GEE_pr','GEE_tmmx','GEE_tmmn','GEE_tmean','tmax','pre','tmin','tmean']])

##pre
x=alldata_pre['pre'].values
y=alldata_pre['GEE_pr'].values

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
ax1 = plt.subplot(1, 2, 1, frameon = False) 
plt.scatter(x, y, c=z, s=10)


##tmin
x=alldata_pre['tmin'].values
y=alldata_pre['GEE_tmmn'].values

xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

ax2 = plt.subplot(1, 2,2, frameon = False) 
plt.scatter(x, y, c=z, s=10)

plt.show()


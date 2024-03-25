# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:17:54 2022

@author: caojuan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 22:21:57 2021

@author: caojuan
"""

import glob
import pandas as pd
import xarray as xr
import os
import rasterio

import numpy as np

# os.chdir(r"Q:\SCI6\results\wheat\wheat0706\Final")   wheat
       
os.chdir(r"Q:\SCI6\results\soybean\FinalTiff1")  
    
filenames = glob.glob('*.tif')

def time_index_from_filenames(filenames):
    '''helper function to create a pandas DatetimeIndex
       Filename example: 20150520_0164.tif'''
    return pd.DatetimeIndex([pd.Timestamp(f[7:11]+'-01-01') for f in filenames]) # wheat [9:13] # soybean[10:14]# maize [9:13] # rice [8:12]

time = xr.Variable('time', time_index_from_filenames(filenames))
chunks1 = {'band': 1,'x': 2924, 'y': 1094}
#da = xr.concat([xr.open_rasterio(f,chunks1).squeeze("band") for f in filenames], dim=time)
#  tranpose and squeeze
# In[]
da = xr.concat([xr.open_rasterio(f,chunks=chunks1 ) for f in filenames], dim=time])
#  tranpose and squeeze
da = da.squeeze("band").transpose("time", "x", "y")
da.name = 'Yield'
da = da.reset_coords(names ='band',drop = True)
da.data[da.data<=0]=np.nan
da = da.rename({'x':'lon','y':'lat'})
# 替换-3.402823e+38 为np.
da.to_netcdf(r'Q:\SCI6\results\datasets\Soybean1982_2015.nc')


# convert the longitude from 0 to 360 to -180 to 180 degrees
# da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby('lon')



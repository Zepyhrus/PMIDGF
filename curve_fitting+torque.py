# -*- coding: utf-8 -*-
# 探究转速、扭矩和温度关系。结果：无明显规律
"""
Created on Mon Jul 24 13:41:09 2017
@author: huangyihong
"""
from datetime import datetime
from scipy.optimize import curve_fit
import time
import dataclassment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
inputfile = 'C:\\Users\\siemens\\Desktop\\Data 2017-07-11\\tor_speed_temp.csv'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%dT%H:%M:%S.%f0')
df = pd.read_csv(inputfile, parse_dates=['time'],\
                 date_parser=dateparse,nrows=500)

df['dtemper'] = df['Temperature'].diff()
def stamp(x):
  x = time.mktime(x.timetuple())
  return x
df['stamp'] = df['time'].map(stamp)
df['dtime'] = df['stamp'].diff()
df['pente'] = df['dtemper']/df['dtime']
df['produit=Speed*Torque'] = df['Speed']*df['Torque']
df['real_time'] = df['stamp']
df['real_T'] = df['Temperature']
df['real_S'] = df['Speed']

## 归一化处理
def norm(df):
  col = []
  for columns in df:
    col.append(columns)
  sta = df.iloc[:,1:9].describe()
  for k in range(1,9):
    df['%s'%col[k]] = (df['%s'%col[k]]-sta['%s'%col[k]].loc['min'])\
    /(sta['%s'%col[k]].loc['max']-sta['%s'%col[k]].loc['min'])
  return df
df_norm = norm(df)
plt.figure(figsize=(14,6))
plt.plot(df_norm['Temperature'])
plt.plot(df_norm['Speed'])
plt.plot(df_norm['Torque'])
plt.legend(loc='upper left')
#plt.ylim(-0.05,0.2)
plt.show()
#step = dataclassment.classment(df,df['stamp'],df['Speed'])
#for i in range(0,len(step)):
#  step[i] = step[i].reset_index([[0, 1, 2, 3]])
## 选取几组数据
#df_sort1 = step[0]
#df_sort2 = step[22]

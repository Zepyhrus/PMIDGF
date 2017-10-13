# HUANG Yihong Edition 2.2
#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
from scipy import interpolate
from nextpow2 import ceillog2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import dataclassment
inputfile = 'C:\\Users\\siemens\\Desktop\\Data 2017-07-11\\timeseries_part.csv'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%dT%H:%M:%S.%f0')
data = pd.read_csv(inputfile, parse_dates=['time'],\
                 date_parser=dateparse)
temps = pd.to_datetime(data['time'])
data = data.round(4)[0:1000] # 小数位数取3
# 将时间转换为时间戳
def stamp(x):
  x = time.mktime(x.timetuple())
  return x
data['stamp'] = data['time'].map(stamp)
data['realtime'] = data['stamp'] - data['stamp'][0]
## 归一化处理
def norma(df):
  col = []
  for columns in df:
    col.append(columns)
  sta = df.iloc[:,1:4].describe()
  for k in range(1,4): #只分析前三列。可修改
    df['%s'%col[k]] = (df['%s'%col[k]]-sta['%s'%col[k]].loc['min'])\
    /(sta['%s'%col[k]].loc['max']-sta['%s'%col[k]].loc['min'])
  return df
data = norma(data)
data = data.round(4)

## 直线拟合
def line_fit(x,y):
  z = np.polyfit(x,y,1)
  return z[0]

## Add time series examine:
data['time_dist'] = data['stamp'].diff()
# 自定义快速傅里叶变换方法
def Myfft(time,value,windowname,dst,noise):
  ## Add time interpolate
  x = time - time[0]
  y = value
  plt.plot(x,y)
  plt.show()
  x_new = np.arange(x[0],x[0]+x[len(x)-1],dst)
  f = interpolate.interp1d(x,y,kind='slinear')
  #'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
  y_new = f(x_new)
  y_new = y_new + noise
  plt.plot(x_new,y_new)
  plt.show()
  # 采样率
  Fs = len(x_new)/x_new[len(x_new)-1]
  print (Fs)
  # 数据长度
  L = len(x_new)
  # 删减数据为fft做准备
  NFFT = int(math.pow(2,ceillog2(L+1)-1))
  use_y = y_new[0:NFFT] 
  windowdata = 0
  if windowname == 'Hamming':
    windowdata = np.hamming(NFFT)
  elif windowname == 'Kaiser': #需要两个输入参数
    windowdata = np.kaiser(NFFT)
  elif windowname == 'Blackman':
    windowdata = np.blackman(NFFT)
  elif windowname == 'Hanning':
    windowdata = np.hanning(NFFT)
  else:
    windowdata = np.ones(1,NFFT)
  use_y = use_y*windowdata
  # fft
  Y = np.fft.fft(use_y,n=NFFT)/NFFT
  Y = 2*abs(Y[0:NFFT/2+1])
  f = np.linspace(0,Fs/2,NFFT/2+1)
  plt.plot(x_new[0:NFFT],use_y[0:NFFT])
  plt.show()
  plt.plot(f,Y)
  plt.show()
  print (NFFT)
  return [f,Y]
step = dataclassment.classment(data,data['stamp'],data['Speed'])
sample = step[0]
## 加噪音
x = np.linspace(0,1,10000)
sinu1 = np.cos(2*np.pi*3000*x-np.pi*30/180) 
sinu2 = 1.5*np.cos(2*np.pi*2000*x-np.pi*30/180)
wave = sinu1[0:210] + sinu2[0:210]
## 用Myff函数
y = Myfft(sample['stamp'],sample['Speed'],'Hanning',0.0001,wave)

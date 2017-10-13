# HUANG Yihong 此Myfft集成了统计量和特征量，生成特征表格
#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
from scipy import interpolate
from nextpow2 import ceillog2
from sklearn.cluster import KMeans
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import dataclassment
import freq_feature
import time_feature
inputfile = 'C:\\Users\\siemens\\Desktop\\Data 2017-07-11\\timeseries_part.csv'
outfile = 'C:\\Users\\siemens\\Desktop\\Data 2017-07-11\\out.csv'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%dT%H:%M:%S.%f0')
data = pd.read_csv(inputfile, parse_dates=['time'],\
                 date_parser=dateparse)
temps = pd.to_datetime(data['time'])
data = data.round(4)[0:1000] # 小数位数取3
def stamp(x):
  x = time.mktime(x.timetuple())
  return x
data['stamp'] = data['time'].map(stamp)
data['realtime'] = data['stamp']
data['realspeed'] = data['Speed']
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
## 生成统计+特征表格，原始Myfft见temp2
def Myfft(df,time,value,windowname,dst,feature,count):
  ## Add time interpolate
  x = time - time[0]
  y = value
  x_new = np.arange(x[0],x[0]+x[len(x)-1],dst)
  f = interpolate.interp1d(x,y,kind='slinear')
  #'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
  y_new = f(x_new)
  # 采样率
  Fs = len(x_new)/x_new[len(x_new)-1]
  # 添加自定义噪声
  nx = np.linspace(0,x[len(x)-1],x[len(x)-1]*1/dst) #100采样率
  sinu1 = 8*np.cos(2*np.pi*0.3*Fs*nx-np.pi*30/180) 
  sinu2 = 5*np.cos(2*np.pi*0.2*Fs*nx-np.pi*30/180)
  noise = sinu1 + sinu2
  y_new = y_new + noise
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
  [Pxx,F] = plt.psd(use_y,NFFT=NFFT,Fs = Fs)
  col = []
  for columns in df:
    col.append(columns)
  sta = df.iloc[:,1:2].describe()  #只分析前一列。可修改
  sta.loc['range'] = sta.loc['max']-sta.loc['min']#极差
  sta.loc['var'] = sta.loc['std']/sta.loc['mean']#变异系数
  sta.loc['dis'] = sta.loc['75%']-sta.loc['25%']#四分位数间距
  sta.loc['peak']=time_feature.peak(use_y)
  sta.loc['rms']=time_feature.rms(use_y)
  sta.loc['pfactor']=time_feature.pfactor(use_y)
  sta.loc['kfactor']=time_feature.kfactor(use_y)
  sta.loc['pulse']=time_feature.pulse(use_y)
  sta.loc['margin']=time_feature.margin(use_y)
  sta.loc['waveform']=time_feature.waveform(use_y)
  sta.loc['sqa']=time_feature.sqa(use_y)
  sta.loc['absma']=time_feature.absma(use_y)
  sta.loc['skewness']=time_feature.skewness(use_y)
  sta.loc['kurtosis']=time_feature.kurtosis(use_y)
  sta.loc['controid_f']=freq_feature.centroid_f(Pxx,F)
  sta.loc['f_variance']=freq_feature.f_variance(Pxx,F)
  sta.loc['ms_f']=freq_feature.ms_f(Pxx,F)
  feature['Speed%d'%(count)] = sta['Speed']
## 工序分类
step = dataclassment.classment(data,data['stamp'],data['Speed'])
for i in range(0,len(step)):
  step[i] = step[i].reset_index([[0, 1, 2, 3]])
## 用Myff函数提取特征到feature
feature = pd.DataFrame()
count = -1
for i in step:
  count = count + 1
  Myfft(i,i['realtime'],i['Speed']-np.mean(i['Speed']),'Hanning',0.01,feature,count)

imp =  feature.loc['peak':'ms_f'] #取出时域、频域特征
imp = imp.T
k = 5  
iteration = 500   
imp_s = 1.0*(imp - imp.mean())/imp.std() 
model = KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)
## 复制到控制台
#model.fit(imp_s)  
#r1 = pd.Series(model.labels_).value_counts()
#r2 = pd.DataFrame(model.cluster_centers_)
#r = pd.concat([r2,r1],axis=1)
#r.columns = list(imp.columns)+['class number']
#data_class = pd.concat([imp,pd.Series(model.labels_,index=imp.index)],axis=1)
#data_class.columns = list(imp.columns)+['class number']

## 分类号写入到step
#for i in range(0,len(step)):
#    step[i]['class'] = data_class.iloc[i]['class number']
## 画图
#plt.figure(figsize=(14,6))
#for i in step:
#    plt.plot(i['time'],i['Speed'])
#plt.show()
## step合并后画图
#result = pd.concat(step)
#result = result.reset_index([[0,1,2,3]])
#plt.figure(figsize=(14,6))
#plt.plot(result['class'])
#plt.show()
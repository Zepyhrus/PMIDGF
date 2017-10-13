# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:39:35 2017
@author: simens
"""
# 依然根据转速划分工序，取扭矩的裕度系数，转速的均方根值、平均值作为信号的特征值。
from scipy import interpolate
from datetime import datetime
from scipy.optimize import curve_fit
import time_feature
import time
import dataclassment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
inputfile = 'C:\\Users\\siemens\\Desktop\\Data 2017-07-11\\tor_speed_temp.csv'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%dT%H:%M:%S.%f0')
df = pd.read_csv(inputfile, parse_dates=['time'],\
                 date_parser=dateparse,nrows=500)

def stamp(x):
  x = time.mktime(x.timetuple())
  return x
df['stamp'] = df['time'].map(stamp)

df['dtemper'] = df['Temperature'].diff()
df['dtime'] = df['stamp'].diff()
df['pente'] = df['dtemper']/df['dtime']
df['real_time'] = df['stamp']
df['real_T'] = df['Temperature']
df['real_S'] = df['Speed']

## 归一化处理
def norm(df):
  col = []
  for columns in df:
    col.append(columns)
  sta = df.iloc[:,1:7].describe()
  for k in range(1,7):
    df['%s'%col[k]] = (df['%s'%col[k]]-sta['%s'%col[k]].loc['min'])\
    /(sta['%s'%col[k]].loc['max']-sta['%s'%col[k]].loc['min'])
  return df
df_norm = norm(df)
plt.figure(figsize=(14,3))
plt.plot(df_norm['Temperature'])
plt.plot(df_norm['Speed'])
plt.plot(df_norm['Torque'])
plt.show()
step = dataclassment.classment(df,df['stamp'],df['Speed'])
for i in range(0,len(step)):
  step[i] = step[i].reset_index([[0, 1, 2, 3]])


# 多项式拟合
def poly_fit(x,y,s):
  z = np.polyfit(x,y,3)
  p = np.poly1d(z)
  print p,',speed =',np.mean(s)
  yvals = p(x)
  plt.plot(x,y,'*')
  plt.plot(x,yvals,'r')
  plt.title('Polynomial Fitting')
  plt.show()
  
## 指数函数拟合
def exp_fit(x,y,s):
  def func(x,c1,c2,c3):
    return c1*np.exp(c2/x+c3)
  popt,pcov = curve_fit(func,x,y)
  c1 = popt[0]
  c2 = popt[1]
  c3 = popt[2]
  yvals = func(x,c1,c2,c3)
  print c1,'*exp(',c2,'/x+',c3,')',',speed =',np.mean(s)
  plt.plot(x,y,'*')
  plt.plot(x,yvals,'r')
  plt.title('Exponential Fitting')
  plt.show()

## 对数函数拟合
def log_fit_cro(x,y,s,i):
  def func(x,c1,c2,c3):
    return c1*np.log(c2*x+c3)
  popt,pcov = curve_fit(func,x,y)
  c1 = popt[0]
  c2 = popt[1]
  c3 = popt[2]
  yvals = func(x,c1,c2,c3)
  speed = np.mean(s)
  return [1,c1,c2,c3,speed,np.mean(i['real_S'])/60]

def log_fit_des(x,y,s,i):
  def func(x,c1,c2,c3,c4):
    return c1*np.log(c2*x+c3)+c4
  popt,pcov = curve_fit(func,x,y)
  c1 = popt[0]
  c2 = popt[1]
  c3 = popt[2]
  c4 = popt[3]
  yvals = func(x,c1,c2,c3,c4)
  speed = np.mean(s)
  print c1,'*log(',c2,'*x+',c3,')','+',c4,',speed =',speed
  plt.plot(x,y,'*')
  plt.plot(x,yvals,'r')
  plt.title('Logarithmic Fitting')
  print (y.corr(yvals))
  plt.show()
  return [0,c1,c2,c3,speed,np.mean(i['real_S'])/60]

# 特征量取扭矩的裕度系数，转速的均值和均方根
def statics(df,feature,count):
    sta = df.iloc[:,1:2].describe() 
    sta.loc['margin'] = time_feature.margin(df['Torque']) 
    sta.loc['rms'] = time_feature.rms(df['Speed']) 
    sta.loc['average'] = sta.loc['mean']
    feature['signal%d'%(count)] = sta.loc['margin':'average']['Speed'] 
    return sta

feature = pd.DataFrame()
count = -1
for i in step:
  count = count + 1
  statics(i,feature,count)
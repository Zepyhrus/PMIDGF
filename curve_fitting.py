# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 13:41:09 2017
@author: huangyihong
"""
#能拟合的工序，对温度进行对数拟合
from scipy import interpolate
from datetime import datetime
from scipy.optimize import curve_fit
import time
import dataclassment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
inputfile = 'C:\\Users\\siemens\\Desktop\\Data 2017-07-11\\timeseries_part.csv'
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%dT%H:%M:%S.%f0')
df = pd.read_csv(inputfile, parse_dates=['time'],\
                 date_parser=dateparse,nrows=1000)

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
  def func(x,c1,c2,c3):
    return c1*np.log(c2/x+c3)
  popt,pcov = curve_fit(func,x,y)
  c1 = popt[0]
  c2 = popt[1]
  c3 = popt[2]
  yvals = func(x,c1,c2,c3)
  speed = np.mean(s)
  print c1,'*log(',c2,'*x+',c3,')',',speed =',speed
  plt.plot(x,y,'*')
  plt.plot(x,yvals,'r')
  plt.title('Logarithmic Fitting')
  print (y.corr(yvals))
  plt.show()
  return [0,c1,c2,c3,speed,np.mean(i['real_S'])/60]

   
def start(step,fea):
    count=0
    for i in step:
        if count == 14 or count == 35 or count == 37 or count == 38:
            count+=1
            continue
        else:
            x = (i['real_time']-i['real_time'][0])/60+1
            y = i['Temperature']
            s = i['Speed']
#            print count
            if len(i)<5:
                count+=1
                continue
            else:
                if i['Temperature'][len(i)-1]-i['Temperature'][0]>0:
                    log_fit_cro(x,y,s,i)
                    fea.loc['cro%d'%(count)]=log_fit_cro(x,y,s,i)
                    count+=1
                else:
                    log_fit_des(x,y,s,i)
                    fea.loc['des%d'%(count)]=log_fit_des(x,y,s,i)
                    count+=1
fea= pd.DataFrame(columns=['cro','a','b','c','speed','Speed'])
start(step,fea)
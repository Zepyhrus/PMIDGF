# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:14:20 2017
@author: simens
"""
#集成了构造的25道振动信号的歪度、均方根、裕度系数，信号特征值：扭矩(裕度系数)、转速（均方根值，均值）
import time_feature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import interpolate
from datetime import datetime
from scipy.optimize import curve_fit
import time
import dataclassment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=[]
x = np.linspace(0,2,1000)
# 现用的是 p = 2000,a = 1,f = 20
# 定义的原始正弦信号
def org(a,p,f1,t=2):
    x = np.linspace(0,t,p)
    sinu = a*np.sin(2*np.pi*f1*x+2*np.pi*np.random.rand())\
    +a*np.sin(2*np.pi*2*f1*x+2*np.pi*np.random.rand())\
    +a*np.sin(2*np.pi*3*f1*x+2*np.pi*np.random.rand())
    return sinu
# 定义的噪声
def noise_(a,p):
    noise = a*(-1+2*np.random.rand(p))
    return noise
# 定义常量信号
def cst(a,sinu):
       return a*max(sinu)
# 定义的冲击信号，p是2f的倍数
def pulse_(a,f,k,t=2,c=50):
    p = c*f
    x = np.linspace(0,t,p)
    y = np.zeros(1000)
    sinu = a*np.sin(2*np.pi*f*x)
    for i in range(p/f*k,p/f*k+p/(4*f)):
        y[i] = sinu[i]  
    for i in range(p/f*k-p/(4*f),p/f*k):
        y[i] = 0.2*sinu[i]
    for i in range(p/f*k+p/(4*f),p/f*k+p/(2*f)):
        y[i] = 0.2*sinu[i]
    for i in range(p/f*k-p/(2*f),p/f*k-p/(4*f)):
        y[i] = 0.1*sinu[i]
    for i in range(p/f*k+p/(2*f),p/f*k+3*p/(4*f)):
        y[i] = 0.1*sinu[i]
    for i in range(p/f*k-3*p/(4*f),p/f*k-p/(2*f)):
        y[i] = 0.05*sinu[i]
    for i in range(p/f*k+3*p/(4*f),p/f*k+p/f):
        y[i] = 0.05*sinu[i]
    for i in range(p/f*k-p/f,p/f*k-3*p/(4*f)):
        y[i] = 0.03*sinu[i]
    for i in range(p/f*k+p/f,p/f*k+p/f+p/(4*f)):
        y[i] = 0.03*sinu[i]
    return y
# 提取特征值
def statics(df,feature,count):
    sta = df.describe()
    sta.loc['rms']=time_feature.rms(df['vib'])
    sta.loc['margin']=time_feature.margin(df['vib'])
    sta.loc['skewness']=time_feature.skewness(df['vib'])
    feature['signal%d'%(count)] = sta.loc['rms':'skewness']['vib']
    return sta
# 画图，并把值保存到df里
def plot_append(x,s,df):
    plt.figure(figsize=(14,3))
    plt.plot(x,s)
    plt.show()
    df.append(pd.DataFrame(s,columns=['vib']))
# 不画图，只把值保存到df里   
def append(s,df):
    df.append(pd.DataFrame(s,columns=['vib']))
plt.figure(figsize=(14,3))  
plt.plot(org(a=1,p=1000,f1=18))  
#pulse = pulse_(a=0.5,p=500,f=25,k=8)
sinu = org(a=1,p=1000,f1=20)
#noise = noise_(a=0.5,p=500)
s0 = org(a=1,p=1000,f1=18)+pulse_(a=0.5,f=18,k=2)+noise_(a=1,p=1000)
append(s0,df=df)
s1 = np.zeros(1000)
append(s1,df=df)
s2 = org(a=1,p=1000,f1=18)+pulse_(a=0.5,f=18,k=2)+noise_(a=1,p=1000)
append(s2,df=df)
s3 = np.zeros(1000)
append(s3,df=df)
s4 = org(a=1,p=1000,f1=24)+pulse_(a=0.5,f=24,k=2)+noise_(a=1,p=1000)
plot_append(x,s4,df=df)
s5 = np.zeros(1000)
append(s5,df=df)
s6 = org(a=1.5,p=1000,f1=33)+pulse_(a=2,f=33,k=2)+noise_(a=1,p=1000)
plot_append(x,s6,df=df)
s7 = np.zeros(1000)
plot_append(x,s7,df=df)
s8 = org(a=1.5,p=1000,f1=33)+pulse_(a=0.5,f=33,k=2)+noise_(a=1,p=1000)
append(s8,df=df)
s9 = org(a=1.5,p=1000,f1=33)+pulse_(a=2,f=33,k=6)+noise_(a=1,p=1000)
append(s9,df=df)
s10 = np.zeros(1000)
append(s10,df=df)
s11 = org(a=1.5,p=1000,f1=33)+pulse_(a=2,f=33,k=8)+noise_(a=1,p=1000)
append(s11,df=df)
s12 = np.zeros(1000)
append(s12,df=df)
s13 = org(a=1.6,p=1000,f1=37)+pulse_(a=0.5,f=37,k=2)+noise_(a=1,p=1000)
append(s13,df=df)
s14 = org(a=1,p=1000,f1=5)+pulse_(a=2,f=5,k=2)+noise_(a=1,p=1000)
plot_append(x,s14,df=df)
s15 = org(a=1,p=1000,f1=20)+pulse_(a=2,f=20,k=2)+noise_(a=1,p=1000)
append(s15,df=df)
s16 = org(a=1.6,p=1000,f1=37)+pulse_(a=0.5,f=37,k=2)+noise_(a=1,p=1000)
plot_append(x,s16,df=df)
s17 = np.zeros(1000)
append(s17,df=df)
s18 = org(a=1,p=1000,f1=20)+pulse_(a=0.5,f=20,k=2)+noise_(a=1,p=1000)+cst(0.1,sinu=sinu)
append(s18,df=df)
s19 = org(a=1,p=1000,f1=20)+pulse_(a=0.5,f=20,k=2)+noise_(a=1,p=1000)+cst(0.1,sinu=sinu)
append(s19,df=df)
s20 = np.zeros(1000)
append(s20,df=df)
s21 = org(a=1,p=1000,f1=20)+pulse_(a=0.5,f=20,k=2)+noise_(a=1,p=1000)+cst(0.1,sinu=sinu)
append(s21,df=df)
s22 = np.zeros(1000)
append(s22,df=df)
s23 = org(a=1,p=1000,f1=20)+pulse_(a=0.5,f=20,k=2)+noise_(a=1,p=1000)+cst(0.1,sinu=sinu)
append(s23,df=df)
s24 = org(a=1,p=1000,f1=20)+pulse_(a=0.5,f=20,k=2)+noise_(a=1,p=1000)+cst(0.1,sinu=sinu)
append(s24,df=df)

feature = pd.DataFrame()
count = -1
for i in df:
  count = count + 1
  sta = statics(i,feature,count)

imp =  feature
imp= imp.T
# 将nan替换为0
for i in range(1,len(imp['margin'])):
    if str(imp['margin'][i]) == 'nan':
        imp['margin'][i] = 0
  
## 
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
df['real_S'] = df['Speed']/60# 转每秒钟

## 归一化处理
def norm(df,a=1,b=7):
  col = []
  for columns in df:
    col.append(columns)
  sta = df.iloc[:,a:b].describe()
  for k in range(a,b):
    df['%s'%col[k]] = (df['%s'%col[k]]-sta['%s'%col[k]].loc['min'])\
    /(sta['%s'%col[k]].loc['max']-sta['%s'%col[k]].loc['min'])
  return df
df_norm = norm(df)
plt.figure(figsize=(14,6))
#plt.plot(df_norm['Temperature'])
plt.plot(df_norm['Speed'])
#plt.plot(df_norm['Torque'])
plt.legend(loc='upper left')
plt.show()
step = dataclassment.classment(df,df['stamp'],df['Speed'])
for i in range(0,len(step)):
  step[i] = step[i].reset_index([[0, 1, 2, 3]])
# 特征量取扭矩的裕度系数，转速的均值和均方根
def statics(df,feature,count):
    sta = df.iloc[:,1:2].describe() 
    sta.loc['margin_Tq'] = time_feature.margin(df['Torque']) 
    sta.loc['rms_S'] = time_feature.rms(df['Speed']) 
    sta.loc['average_S'] = sta.loc['mean']
    feature['signal%d'%(count)] = sta.loc['margin_Tq':'average_S']['Speed'] 
    return sta

feature = pd.DataFrame()
count = -1
for i in step:
  count = count + 1
  statics(i,feature,count)
  

fea = feature.T
imp = imp.reset_index([[0, 1, 2, 3]])
fea = fea.reset_index([[0, 1, 2, 3]])
result = pd.concat([fea, imp], axis=1)
imp = result
imp = norm(imp,a=0,b=5)

k = 4  
iteration = 500   
imp_s = 1.0*(imp - imp.mean())/imp.std() 
model = KMeans(n_clusters=k,n_jobs=4,max_iter=iteration)

## 注意：以下代码复制到控制台运行，进行分类！
#model.fit(imp_s)  
#r1 = pd.Series(model.labels_).value_counts()
#r2 = pd.DataFrame(model.cluster_centers_)
#r = pd.concat([r2,r1],axis=1)
#r.columns = list(imp.columns)+['class number']
#data_class = pd.concat([imp,pd.Series(model.labels_,index=imp.index)],axis=1)
#data_class.columns = list(imp.columns)+['class number']



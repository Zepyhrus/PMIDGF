# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 14:22:54 2017
@author: huang
"""
# HUANG Yihong Edition 2.2
#!/usr/bin/env python
from __future__ import print_function
from sklearn.cluster import KMeans
import numpy as np         
# 峰值（不是最大值）
def peak(x):
  l = len(x)
  peak=((max(x[0:l/2])-min(x[0:l/2]))/2+(max(x[l/2+1:l])-min(x[l/2+1:l]))/2)/2
  return peak

# 均方根值
def rms(x):
  rms = np.sqrt(sum(x**2)/len(x))
  return rms

# 峰值因子
def pfactor(x):
  pfactor = peak(x)/rms(x)
  return pfactor

# 峭度因子 kurtosis factor
def kfactor(x):
  kfactor = sum(((x-np.average(x))**4))/((rms(x)**4)*len(x))
  return kfactor

# 脉冲因子
def pulse(x):
  pulse = (peak(x)*len(x))/sum(abs(x))
  return pulse

# 裕度系数
def margin(x):
  margin = max(x)/(sum(np.sqrt(abs(x)))/len(x))**2
  return margin

# 裕度系数自创改进版
def margin2(x):
  data = sorted(x,reverse=True)  
  max_ = np.average(data[0:10])
  margin = max_/(sum(np.sqrt(abs(x)))/len(x))**2
  return margin

# 波形因子
def waveform(x):
  waveform = (rms(x)*len(x))/sum(abs(x))
  return waveform

# 方根幅值
def sqa(x):
  sqa = (sum(np.sqrt(abs(x)))/len(x))**2
  return sqa

# 绝对平均幅值
def absma(x):
  absma = sum(abs(x))/len(x)
  return absma

# 歪度
def skewness(x):
  skewness = sum(x**3)/len(x)
  return skewness

# 峭度
def kurtosis(x):
  kurtosis = sum(x**4)/len(x)
  return kurtosis
# 打印所有时域特征
def printfeature(x):
    print ('\n','时域特征','\n','峰值:',peak(x),'\n','均方根值:',rms(x),'\n','峰值因子:',peak(x)/rms(x),\
           '\n','峭度因子:',kfactor(x),'\n','脉冲因子:',pulse(x),'\n','裕度系数:',\
           margin(x),'\n','波形因子:',waveform(x),'\n','方根幅值:',sqa(x),'\n',\
           '绝对平均幅值:',absma(x),'\n','歪度:',skewness(x),'\n','峭度:',kurtosis(x),'\n')
    
def feature(x):
    print ('均方根值:',rms(x),'\n','裕度系数:',margin2(x),'\n','歪度:',skewness(x),'\n')
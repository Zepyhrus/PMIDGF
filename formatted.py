# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:57:29 2017
@author: HUANG Yihong
"""
#使raw data 转换成 formatteddata
import matplotlib.pyplot as plt


## 空值填补
def fillvide(df):
  df = df.fillna(method='pad')   #空值用下一个值填补,遗留列尾空值
  df = df.fillna(method='bfill')  #空值用上一个值填补,处理列尾空值
  return df


## 处理Unicode数据（不含时间列）
def delet(df):
  for columns in df:
    if columns=='time': #time也是unicode需忽略
      continue
    if all (type(x)==unicode for x in df['%s'%columns]): #删除Unicode列
      del df['%s'%columns]
      continue
    for i in range(0,len(df['%s'%columns])): #处理零散Unicode值
      if type(df['%s'%columns][i])==unicode and i==0:#第一行为unicode时，依次向
        for j in range(1,len(df['%s'%columns])):     #下用第一个非Unicode来替换
          if type(df['%s'%columns][j]) != unicode:
            df['%s'%columns][i] == df['%s'%columns][j]
          else:
            continue
      if type(df['%s'%columns][i])==unicode and i>0:#零散Unicode值用上个值替换
        df['%s'%columns][i] = df['%s'%columns][i-1] 
  return df


## 对分类后的数据：箱型法画出异常值，处理异常值
def box(df,last=4):
  col=[]
  for columns in df:
    col.append(columns)
  for k in range(1,last):# 只处理前三列：转速、电流、扭矩。可据需求修改
    plt.figure()
    p = df.iloc[:,k:k+1].boxplot(return_type='dict')
    x = p['fliers'][0].get_xdata()
    y = p['fliers'][0].get_ydata()
    y.sort()
    # 在箱型图异常点上标数值
    for i in range(len(x)):
      if i>0:
        plt.annotate(y[i],xy = (x[i],y[i]), xytext=(x[i]+0.05-0.8/
                     (y[i]-y[i-1]),y[i]))
      else:
        plt.annotate(y[i],xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))
    plt.show() 
    # 异常值在列首：下个值修正。在列尾：上个值修正。在列中：相邻均值修正
    for i in y:
      j = round(i,3)
      location = list(df['%s'%col[k]]).index(j)
      if location == 0:
        df['%s'%col[k]].iloc[location] = df['%s'%col[k]].iloc[location+1]
      elif location == len(df['%s'%col[k]]) - 1:
         df['%s'%col[k]].iloc[location] = df['%s'%col[k]].iloc[location-1]
      else:
        df['%s'%col[k]].iloc[location] = ((df['%s'%col[k]].iloc[location+1]
      + df['%s'%col[k]].iloc[location-1])/2).round(3)
  return df
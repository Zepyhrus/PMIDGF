# -*- coding:utf-8 -*-
# HUANG Yihong Edition 2.2
#!/usr/bin/env python
"""
Created on Mon Aug 07 14:58:28 2017
@author: siemens
"""
import numpy as np


def line_fit(x,y):
  z = np.polyfit(x,y,1)
  return z[0]

# 数据分类：单个跳变忽略。连续两跳变忽略，然后开始新一类。连续三跳变视为一新类。
def classment(data,x,y,c=10):
  step = []
  count = 0
  a = 0 
  crt = c  #阈值，根据需要进行修改
  for i in range(1,len(x)):
    pente=abs(line_fit(x.iloc[i-1:i+2],y.iloc[i-1:i+2]))
    if pente >crt:
      if abs(line_fit(x.iloc[i:i+3],y.iloc[i:i+3]))<crt:
        continue
      else:
        if abs(line_fit(x.iloc[i+1:i+4],y.iloc[i+1:i+4]))>crt:
          continue
        else:
          step.append(data.iloc[a:i+1])
          count = count + 1
          a = i + 1
    else:
          continue  
  step.append(data.iloc[a:])
  return step

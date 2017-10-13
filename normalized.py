# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:47:47 2017
@author: HUANG Yihong
"""
# 最小-最大规范化  
def norma(df,last=4):
  col = []
  for columns in df:
    col.append(columns)
  sta = df.iloc[:,1:last].describe()
  for k in range(1,last): #只分析前三列。可修改
    df['%s'%col[k]] = (df['%s'%col[k]]-sta['%s'%col[k]].loc['min'])\
    /(sta['%s'%col[k]].loc['max']-sta['%s'%col[k]].loc['min'])
  return df
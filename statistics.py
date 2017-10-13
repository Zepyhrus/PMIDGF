# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:10:02 2017
@author: HUANG Yihong
"""
# 生成统计表格feature。count计工序用。
def fea(df,count,feature,last=2):
  col = []
  for columns in df:
    col.append(columns)
  sta = df.iloc[:,1:2].describe()  #只分析前一列。可修改
  sta.loc['range'] = sta.loc['max']-sta.loc['min']#极差
  sta.loc['var'] = sta.loc['std']/sta.loc['mean']#变异系数
  sta.loc['dis'] = sta.loc['75%']-sta.loc['25%']#四分位数间距
  for k in range(1,last): #需要分析的列数。可修改
    feature['%s%d'%(col[k],count)] = sta['%s'%col[k]]
  return feature
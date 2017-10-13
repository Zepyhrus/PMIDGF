# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:25:51 2017
@author: simens
"""
#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 重心频率
def centroid_f(Pxx,f):
    centroid_f = sum(Pxx*f)/sum(Pxx)
    return centroid_f
# 频率方差
def f_variance(Pxx,f):
    f_variance = sum((f-centroid_f(Pxx,f))**2*Pxx)/sum(Pxx)
    return f_variance
# 均方频率
def ms_f(Pxx,f):
    ms_f = sum(f**2*Pxx)/sum(Pxx)
    return ms_f
# 打印频域特征值
def printfeature(Pxx,f):
    print '频域特征','\n','重心频率:',centroid_f(Pxx,f),'\n','频率方差:',\
    f_variance(Pxx,f),'\n','均方频率:',ms_f(Pxx,f)
a =1

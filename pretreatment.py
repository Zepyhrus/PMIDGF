# HUANG Yihong Edition 2.1（用自建函数）
#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function
import pandas as pd
import formatted
inputfile = 'C:\Users\simens\Desktop\Data 2017-07-11\cnc_data.xls'
outputfile = 'C:\Users\simens\Desktop\Data 2017-07-11\cnc_data_done.xls'
data = pd.read_excel(inputfile, header=0)
data = data[data['program'] == 1.8]
data = data.reset_index([[0, 1, 2, 3, 4, 5]])
temps = pd.to_datetime(data['time'])
data = data.round(3) # 小数位数取3
formatted.fillvide(data) # 处理缺失值
formatted.delet(data) # 剔除非数字数据
## 按流程分为18种工况
step = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
temp = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
step[0] = data[(data['time']>='2017-03-07 15:00:00')&
             (data['time']<='2017-03-07 15:03:59')]
temp[0] = (temps[list(data['time']).index('2017-03-07 15:03:59')] 
- temps[0]).seconds
step[1] = data[(data['time']>='2017-03-07 15:04:00')&
             (data['time']<='2017-03-07 15:04:59')]
temp[1] = (temps[list(data['time']).index('2017-03-07 15:04:59')] 
- temps[0]).seconds
step[2] = data[(data['time']>='2017-03-07 15:05:00')&
             (data['time']<='2017-03-07 15:07:01')]
temp[2] = (temps[list(data['time']).index('2017-03-07 15:07:01')] 
- temps[0]).seconds
step[3] = data[(data['time']>='2017-03-07 15:07:02')&
             (data['time']<='2017-03-07 15:07:57')]
temp[3] = (temps[list(data['time']).index('2017-03-07 15:07:57')] 
- temps[0]).seconds
step[4] = data[(data['time']>='2017-03-07 15:07:58')&
             (data['time']<='2017-03-07 15:08:22')]
temp[4] = (temps[list(data['time']).index('2017-03-07 15:08:22')] 
- temps[0]).seconds
step[5] = data[(data['time']>='2017-03-07 15:08:23')&
             (data['time']<='2017-03-07 15:09:18')]
temp[5] = (temps[list(data['time']).index('2017-03-07 15:09:18')] 
- temps[0]).seconds
step[6] = data[(data['time']>='2017-03-07 15:09:19')&
             (data['time']<='2017-03-07 15:09:37')]
temp[6] = (temps[list(data['time']).index('2017-03-07 15:09:37')] 
- temps[0]).seconds
step[7] = data[(data['time']>='2017-03-07 15:09:38')&
             (data['time']<='2017-03-07 15:10:03')]
temp[7] = (temps[list(data['time']).index('2017-03-07 15:10:03')] 
- temps[0]).seconds
step[8] = data[(data['time']>='2017-03-07 15:10:04')&
             (data['time']<='2017-03-07 15:10:19')]
temp[8] = (temps[list(data['time']).index('2017-03-07 15:10:19')] 
- temps[0]).seconds
step[9] = data[(data['time']>='2017-03-07 15:10:20')&
             (data['time']<='2017-03-07 15:10:47')]
temp[9] = (temps[list(data['time']).index('2017-03-07 15:10:47')] 
- temps[0]).seconds
step[10] = data[(data['time']>='2017-03-07 15:10:48')&
             (data['time']<='2017-03-07 15:11:23')]
temp[10] = (temps[list(data['time']).index('2017-03-07 15:11:23')] 
- temps[0]).seconds
step[11] = data[(data['time']>='2017-03-07 15:11:24')&
             (data['time']<='2017-03-07 15:11:44')]
temp[11] = (temps[list(data['time']).index('2017-03-07 15:11:44')] 
- temps[0]).seconds
step[12] = data[(data['time']>='2017-03-07 15:11:45')&
             (data['time']<='2017-03-07 15:12:15')]
temp[12] = (temps[list(data['time']).index('2017-03-07 15:12:15')] 
- temps[0]).seconds
step[13] = data[(data['time']>='2017-03-07 15:12:16')&
             (data['time']<='2017-03-07 15:12:41')]
temp[13] = (temps[list(data['time']).index('2017-03-07 15:12:41')] 
- temps[0]).seconds
step[14] = data[(data['time']>='2017-03-07 15:12:42')&
             (data['time']<='2017-03-07 15:13:09')]
temp[14] = (temps[list(data['time']).index('2017-03-07 15:13:09')] 
- temps[0]).seconds
step[15] = data[(data['time']>='2017-03-07 15:13:10')&
             (data['time']<='2017-03-07 15:13:30')]
temp[15] = (temps[list(data['time']).index('2017-03-07 15:13:30')] 
- temps[0]).seconds
step[16] = data[(data['time']>='2017-03-07 15:13:31')&
             (data['time']<='2017-03-07 15:13:53')]
temp[16] = (temps[list(data['time']).index('2017-03-07 15:13:53')] 
- temps[0]).seconds
step[17] = data[(data['time']>='2017-03-07 15:13:54')&
             (data['time']<='2017-03-07 15:14:08')]
temp[17] = (temps[list(data['time']).index('2017-03-07 15:14:08')] 
- temps[0]).seconds
# 数据清洗
for i in step:
  formatted.box(i) 

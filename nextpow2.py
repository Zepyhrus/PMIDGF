# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 15:14:19 2017
@author: siemens
"""
import ctypes as ct
class FloatBits(ct.Structure):
    _fields_ = [
        ('M', ct.c_uint, 23),
        ('E', ct.c_uint, 8),
        ('S', ct.c_uint, 1)
    ]

class Float(ct.Union):
    _anonymous_ = ('bits', )
    _fields_ = [
        ('value',   ct.c_float),
        ('bits',    FloatBits)
    ]
# 相当于matlab中的nextpow2
def ceillog2(x):
    if x < 0:
        x = -x
    if x == 0:
        return 0
    d = Float()
    d.value = x
    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1

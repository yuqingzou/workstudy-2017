# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:20:35 2017

@author: hannah.li
"""
import pandas as pd
s=0
n=10
df = pd.read_csv('/home/hannah.li/Yuqing/Q_15_test_mw.csv')
for i in range(int(len(df)/10)):
    print(df[s:s+n])
    s=s+n
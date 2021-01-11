import tushare as ts
import pandas as pd
import os
import time
import glob
import numpy as np


code='510050'
date1='2000-12-01'
date2='2020-12-31'
filename='..\\data\\'
length=-1 

df = ts.get_hist_data(code, start=date1, end=date2)
df1 = pd.DataFrame(df)


df1 = df1[['close','open', 'high', 'low', 'volume', 'price_change','p_change']]
df1 = df1.sort_values(by='date')
print('共有%s天数据' % len(df1))
if length == -1:
    path = code + '.csv'
    df1.to_csv(os.path.join(filename, path))
else:
    if len(df1) >= length:
        path = code + '.csv'
        df1.to_csv(os.path.join(filename, path))
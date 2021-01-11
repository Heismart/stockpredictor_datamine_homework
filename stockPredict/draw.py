from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

data = read_csv('pred.csv',header=None,index_col=None)

cnt=0
for i in range(1,len(data[0])):
    if(data[0][i]>data[0][i-1] and data[1][i]>data[1][i-1]):
        cnt=cnt+1
    elif(data[0][i]<data[0][i-1] and data[1][i]<data[1][i-1]):
        cnt=cnt+1



print('accuracy:%.2f'%(cnt/len(data[0])))



plt.plot(range(1,len(data[0])+1),data[0],label='pred')
plt.plot(range(1,len(data[1])+1),data[1],label='true')

plt.legend()
plt.show()


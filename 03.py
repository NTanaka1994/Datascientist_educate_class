#必要
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series,DataFrame
import pandas as pd
import time

#可視化
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

#機械学習
import sklearn

#少数第三位

#答え
import csv
f=open("bank/bank-full.csv","r")
reader=csv.reader(f)
tmp=[]
x=[]

j=0
name=["age","balance","day","duration","campaign","pdays","previous"]
num=[]
for row in reader:
    if j==0:
        for i in range(len(row)):
            for k in range(len(name)):
                if row[i]==name[k]:
                    num.append(i)
        j=1
    else:
        for i in range(len(num)):
            tmp.append(float(row[num[i]]))
        x.append(tmp)
        tmp=[]
print("レコード数"+str(len(x)))
x2=np.array(x)
for i in range(len(name)):
    print(name[i]+"の最大値="+str(max(x2[:,i])))
    print(name[i]+"の最小値="+str(min(x2[:,i])))
    print(name[i]+"の標準偏差"+str(np.std(x2[:,i])))


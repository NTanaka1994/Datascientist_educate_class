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
f=open("abalone.csv","r")
reader=csv.reader(f)
tmp=[]
x=[]
y=[]
for row in reader:
    for i in range(len(row)):
        if i==len(row)-1:
            y.append(row[i])
        elif i!=0:
            tmp.append(row[i])
    x.append(tmp)
    tmp=[]
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(x, y)
a = reg.coef_
b = reg.intercept_
for i in range(len(a)):
    print("因子"+str(i+1)+"="+str(a[i]))
print("切片"+str(b))

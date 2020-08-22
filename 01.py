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


from sklearn.datasets import load_digits

digits=load_digits()
plt.figure(figsize=(20,5))
for label,img in zip(digits.target[:10],digits.images[:10]):
    plt.subplot(1,10,label+1)
    plt.axis('off')
    plt.imshow(img,cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title("Number:{0}".format(label))
plt.show()

#答え
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
model=SVC(gamma=0.001,C=1)
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
m=confusion_matrix(y_test,y_pred)
print("confusion matrix:\n{}".format(m))

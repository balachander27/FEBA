'''
Authors: Balachander S, Prahalad Srinivas C G, Yogesh Chandra Singh Samant, B Varshin Hariharan
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

class FeatureClassifier:
  def __init__(self,reqAcc=0.80,classifier='DesicionTree',bias0=0,bias1=0,depth=None):
    self.featureClassifiers=[]
    self.reqAcc=reqAcc
    self.indexLs=[]
    self.flag=0
    self.bias0=bias0
    self.bias1=bias1
    self.depth=depth
    self.classifier=classifier
    self.dic={'DesicionTree':0,'LinearRegression':1,'SVM':2,'LogisticRegression':3}
  def finIndex(self):
    for i in range(len(self.featureClassifiers)):
      if self.featureClassifiers[i][1] < self.reqAcc:
        return i
      self.indexLs.append(self.featureClassifiers[i][2])
    self.flag=1
    return i
  def fit(self,x,y):
    for i in range(len(x[0])):
      clf=[DecisionTreeClassifier(),LinearRegression(),SVC(), LogisticRegression()][self.dic[self.classifier]]
      X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
      clf.fit(X_train[:,i:i+1],y_train)
      self.featureClassifiers.append((clf,accuracy_score(y_test,np.round(clf.predict(X_test[:,i:i+1]))),i))
    self.featureClassifiers.sort(key=lambda x:x[1],reverse=True)
    index=self.finIndex()
    if self.flag==0:
      self.featureClassifiers=self.featureClassifiers[:index]
    else:
      pass
    return
  def predict(self,x):
    class0=0
    class1=1
    yPred=[]
    for i in range(len(x)):
      class0=0
      class1=0
      for j in range(len(self.indexLs)):
        pred=self.featureClassifiers[j][0].predict([[x[i][self.indexLs[j]]]])
        if pred == 0:
          class0+=self.featureClassifiers[j][1]+self.bias0
        else:
          class1+=self.featureClassifiers[j][1]+self.bias1
      yPred.append(1 if class1>class0 else 0)
    return yPred
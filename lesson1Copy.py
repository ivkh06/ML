    # -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:08:30 2017

@author: s2286066
"""

import sklearn 
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
import csv
import pandas as pd
from sklearn import tree
import random
from sklearn import svm, datasets
#from sklearn.model_selection import train_test_split
from sklearn import cross_validation

def split_data(x,y, split_rate=0.75):    
    randomize = np.arange(len(x))
    np.random.shuffle(randomize)
    x = x[randomize]
    y = y[randomize]   
    split = int(split_rate*points)
    X_train = x[0:split]
    X_test  = x[split:]
    y_train = y[0:split]
    y_test  = y[split:]
    return features_train, labels_train, features_test, labels_test

################# Naive Gausse
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf_pf.predict([[-0.8, -1]]))
[1]

#########################     SVM start   -------

###########   1. importing Terrain  data
################## import terrain data############################
features_train, labels_train, features_test, labels_test = makeTerrainData()
acuracy_test()

###########   2.
################## import iris data set############################
iris = datasets.load_iris()
x=iris.data
y=iris.target
# this my own 
#features_train, labels_train, features_test, labels_test=split_data(x,y, split_rate=0.75)

# decision tree            can be skipped
clf = tree.DecisionTreeClassifier(min_samples_split=22)
clf.fit(features_train, labels_train)
pred=clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
accuracy_score(labels_test,pred)
print (acc)
################  cross sector validation ##################

# this correct
features_train, features_test, labels_train,  labels_test = cross_validation.train_test_split(
   iris.data, iris.target, test_size=0.8, random_state=0)
acuracy_test()

###########   3.
################## German credt data ############################

df= pd.read_csv('C:\\machine\\Data\\'+'german_credit.csv')
x=df[df.columns[1:]]
y=df[df.columns[0]]

features_train, features_test, labels_train, labels_test  = cross_validation.train_test_split(
   x, y, test_size=0.2, random_state=0)
acuracy_test()

#  Decision tree
clf = tree.DecisionTreeClassifier(min_samples_split=30)
clf.fit(features_train, labels_train)

pred=clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
accuracy_score(labels_test,pred)
print (acc)

###########   4.
################## LATAM DATA  and Mexico############################
def read_header(file_name):
    ''' return header of CSV file'''    
    fp = open(file_name)
    line =fp.readline()
    line=line.strip('\n') 
    header=line.split(',')
    fp.close()
    return header  
dirname='U:\\Python\\'
name='Mexico2.csv'
fields=read_header(dirname+name)
tp = pd.read_csv(dirname+name, skipinitialspace=True,  
                       encoding="ISO-8859-1", iterator=True, chunksize=1000)  # gives TextFileReader, which is iterable with chunks of 1000 rows.
df_Mexico = pd.concat(tp, ignore_index=True) 
df_Mexico.info()
#useful_col=df_Mexico.columns.values[23:53] 
### [23:46]  drivers are located in these columns
drivers=df_Mexico.columns.values[23:46] 
df_Mex_drivers=df_Mexico[drivers]
del df_Mex_drivers[df_Mexico.columns.values[30]]


df_Mex_drivers.info()
Mex_IG=df_Mexico['Model Generated IG']
### creating training  and testing samples on Mexico Data
features_train, features_test, labels_train, labels_test  = cross_validation.train_test_split(
   df_Mex_drivers, Mex_IG, test_size=0.2, random_state=0)

acuracy_test()
############  Using decision tree
clf = tree.DecisionTreeClassifier(min_samples_split=22)
clf.fit(features_train, labels_train)

pred=clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
accuracy_score(labels_test,pred)
print (acc)
##### end of fitting

def transition_matrix(ig1,ig2):
    ig1=np.array(ig1)
    ig2=np.array(ig2)
    listt=[21,27,40,60,65,70,73,75,77,80,83,85,87,90,95,98]
    N=len (listt)
    m = (N,N)
    m=np.zeros(m)
    df=pd.DataFrame(data=m, columns=listt, index=listt)
    for i  in range (len (ig1)):
        df.loc[ig1[i], ig2[i]]+=1
    return df

df=transition_matrix(labels_test,pred)
df.to_csv('U:\\Python\\view.csv', header=True)
print (df)

# Loading LAGRT data
name='LAGRT_full.csv'
fields=read_header(dirname+name)
tp = pd.read_csv(dirname+name, skipinitialspace=True,  
                       encoding="ISO-8859-1", iterator=True, chunksize=1000)  # gives TextFileReader, which is iterable with chunks of 1000 rows.
df = pd.concat(tp, ignore_index=True) 
df.info()
# selecting columns with drivers
listt=df.columns.values[23:53]
#col_list=df_lagrt.columns.values   # not working
col_list=df.columns.values  
useful_names=list(col_list[18:25])+list(col_list[41:56]) 
df_lagrt=df[useful_names]
df_lagrt.info()
lagrt_ig=df['Model_IG']   # defining IG

# splitting data
features_train, features_test, labels_train, labels_test  = cross_validation.train_test_split(
   df_lagrt, lagrt_ig, test_size=0.2, random_state=0)
acuracy_test()


# testing data with different   min_samples_split
res=[]
for j in range (2,60):
    clf = tree.DecisionTreeClassifier(min_samples_split=j)
    clf.fit(features_train, labels_train)    
    pred=clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)    
    res.append(acc)
plt.plot(res)

#training model on  LAGRT data data and testing  them on Mexico
features_train, features_test, labels_train, labels_test=df_lagrt, df_Mex_drivers, lagrt_ig, Mex_IG 
res=[]
for j in range (2,60):
    clf = tree.DecisionTreeClassifier(min_samples_split=j)
    clf.fit(features_train, labels_train)    
    pred=clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)    
    res.append(acc)
plt.plot(res)

###########   5.
################## Enron e-mails data ############################

import email_preprocess as pp
features_train, features_test, labels_train, labels_test=pp.preprocess()
acuracy_test()



###########   6.
################## Titanic data ############################
dname='C:\\machine\Data\\'
df_all= pd.read_csv(dname+'titanic_train.csv')


df_all[df_all.Sex=='male']=1
df_all[df_all.Sex=='female']=2

df_=df_all[df_all.Age>=0]
#df_.info()
features_train=df_all[['Pclass', 'Sex','Fare','Age']]
#features_train.head(10)

labels_train=df_['Survived']

dname='C:\\machine\Data\\'
df_all= pd.read_csv(dname+'titanic_test.csv')

df_all[df_all.Sex=='male']=1
df_all[df_all.Sex=='female']=2
df_=df_all[df_all.Age>=0]
df_.info()
features_test=df_all[['Pclass', 'Sex','Fare','Age']]


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)  
pred=clf.predict(features_test)

#from sklearn.metrics import accuracy_score    do not have reported data 
#acc = accuracy_score(pred, labels_test)
#print (acc)

################################## TESTING  ####################################
#########       testing with different methods

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#     testing support vector machines                       #############################

acuracy_test()

def acuracy_test(): 
    clf = SVC(kernel="linear")
    clf.fit(features_train,labels_train)  
    pred=clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print ('support vector machines', round(acc,3) )
    
    
    ###########  Bayes                                        #############################
    clf = GaussianNB()
    clf.fit(features_train,labels_train)  
    pred=clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print ('Bayes', round(acc,3) )
    
    #  Decision tree                                       #############################
    clf = tree.DecisionTreeClassifier(min_samples_split=22)
    clf.fit(features_train, labels_train)
    
    pred=clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    accuracy_score(labels_test,pred)
    print ('Decision tree', round(acc,3) )


#   Adaboost
# The five boxing wizards jump quickly.





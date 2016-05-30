__author__ = 'Karan Chopra'

'''
#Use this snippet to create the feature matrix
# the matrix is stored in .mat file format!
'''
import pandas as pd
import numpy as np
import scipy.io as scio
from sklearn.utils import shuffle
import os

topics=pd.read_csv('all-topics-strings.lc.txt',)
topics_vec=np.matrix(topics)
#print topics_vec


vocabdic={}
'''top=topics_vec[x,0]
top=str(top).strip()
file=open('Vocab'+top+'.csv','r')'''
file=open('Vocab_dir.csv','r')
data=file.readline().replace('\r\n','')
data=data.split(',')
print len(data)
print data

for x in range(0,len(data)):
    vocabdic.update({data[x]:x})
#print vocabdic
print len(vocabdic)




''' ### Train The Vector Machine ### '''
xfiles= os.listdir('F:\Learning Practice\Pyhton Practice\Reuters\documents')

X_train=None
Y_train=None
for f in xfiles:
    fl=open('.\documents1\\'+str(f),'r')
    data=fl.readline().replace('\r\n','')
    data=data.split(',')
    Yx=fl.readline().replace('\r\n','').replace('[','').replace('.]','').strip()
    Yx=Yx.split(',')
    Yx=map(float,Yx)
    Yx=map(int,Yx)
    Yx=np.matrix(Yx)
    #print Yx
    wx=[]

    for x in range(0,len(data)):
        if vocabdic.get(data[x])!=None:
            wx+=[vocabdic.get(data[x])]
    Xx=np.zeros((len(vocabdic),1))
    if len(wx)>1:
        Xx[np.matrix(wx),0]=1
    #print Xx
    if X_train==None:
        X_train=np.matrix(Xx.reshape((1,Xx.shape[0])))
    else:
        X_train=np.append(X_train,Xx.reshape((1,Xx.shape[0])),axis=0)

    if Y_train==None:
        Y_train=np.matrix(Yx)

    else:
        #print Y_train.shape
        #print Yx.shape
        Y_train=np.append(Y_train,Yx,axis=0)
    print '%s - %s'% ('Completed Parsing File',f)


print Y_train.shape
print(X_train).shape

X_train,Y_train=shuffle(X_train,Y_train)


X_test=X_train[0:int(X_train.shape[0]*0.3),:]
X_train=X_train[int(X_train.shape[0]*0.3):,:]

Y_test=Y_train[0:int(Y_train.shape[0]*0.3),:]
Y_train=Y_train[int(Y_train.shape[0]*0.3):,:]

scio.savemat('X_train.mat',{'X_train':X_train})
scio.savemat('X_test.mat',{'X_test':X_test})
scio.savemat('Y_train.mat',{'Y_train':Y_train})
scio.savemat('Y_test.mat',{'Y_test':Y_test})
print 'Created the feature matrix file with Train set to Test Set ratio - 70/30'

__author__ = 'Karan Chopra'
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import scipy.io as scio
import numpy as np
data=scio.loadmat('X_train.mat')
X_train=np.matrix(data['X_train'])
data=scio.loadmat('X_test.mat')
X_test=np.matrix(data['X_test'])
data=scio.loadmat('Y_train.mat')
Y_train=np.matrix(data['Y_train'])
data=scio.loadmat('Y_test.mat')
Y_test=np.matrix(data['Y_test'])


print X_test.shape
print X_train.shape
print Y_test.shape
print Y_train.shape

LSVM= OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(X_train, Y_train )

pred=LSVM.predict(X_test)
#print Y_test
#print np.where(Y_test==pred)[0]
count=0
#print len(np.where(Y_test[50,:]==pred[50,:].reshape((1,pred[50,:].size))))
for x in range(0,Y_test.shape[0]):
    count+=len(np.where(Y_test[x,:]==pred[x,:].reshape((1,pred[x,:].size)))[1])
print np.where(Y_test[x,:]==pred[x,:].reshape((1,pred[x,:].size)))
#print Y_test[50,:].shape
print 'Testing of data is finished!'
print 'The accuracuy on the test set is- '+str(len(np.where(Y_test==pred)[0])*100.0/(Y_test.shape[0]*Y_test.shape[1]))

exit()
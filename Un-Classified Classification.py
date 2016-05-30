__author__ = 'Karan Chopra'

from Create_Vocabolary import text_normalize
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import scipy.io as scio

body='''The ruling Liberal Democratic Party's
(LDP) setback in Sunday's nationwide local elections may force
the government to water down its controversial proposal for a
five pct sales tax and undermine its commitment to stimulating
the economy, private economists said.
    The LDP's failure to win seats in some crucial local
constituencies will weaken the government's ability to push
through its tax plan, and without a compromise tax proposal the
budget for fiscal 1987/88 ending March 31 is unlikely to be
passed soon, they said.
    Without the budget, the government would also be
hard-pressed to come up with an effective package to stimulate
the economy as pledged at Group of Seven meetings in Paris in
late February and in Washington last week, they said.
    Opposition protests against the sales tax have stalled
parliamentary debate on the budget for weeks and forced the
government to enact a stop-gap 1987/88 budget that began early
this month.
    "The LDP's election setback will have an enormous impact on
the already faltering economy," said Johsen Takahashi, chief
economist at Mitsubishi Research Institute.
    Takahashi said that behind the LDP's poor showing was
public discontent with the government's high-handed push for
tax reform and its lack of effective policies to cope with
economic woes caused by the yen's appreciation.
    "This explains why the LDP failed to regain governorships in
the most hotly contested constituencies of Fukuoka and
Hokkaido, where the shipbuilding and steel industries are
suffering heavily from the yen's extended rise," he said.
    Takahashi said the government should delay introduction of
the sales tax for one or two years beyond its original starting
date of January 1988 and implement tax cuts now.
    Sumitomo Bank Ltd chief economist Masahiko Koido also said
he favours watering down the proposed sales tax while
suggesting the government boost public works spending by
modifying its tight fiscal policies.
    "The local election results were a signal the economy now
needs government action to take clear-cut fiscal measures," he
said, adding that such moves would help the world economy as
well as Japan's.
    For the last five years or so, the government has stuck to
a tight fiscal policy in a bid to halt the issue of deficit
financing bonds by fiscal 1990/91, economists said.
    If the LDP election setback leads to a scaled down sales
tax proposal, the government would have to find other revenue
sources to help finance the planned tax cuts and a package of
measures to stimulate the economy, the economists said.
    Koido said the government could raise additional revenue by
selling shares in public corporations such as Nippon Telegraph
and Telephone Corp. But it should issue additional bonds to
ensure a more stable source of funds, he said.
    "It may run counter to its avowed policy of balancing the
budget, but it can do so on a short-term basis," he said.
    Takahashi of Mitsubishi Research also agreed with the need
for the government to float more bonds to raise funds needed
for economic expansion.
    He said additional government borrowing would place no
burden on the capital market because it has amassed huge excess
funds and government bond prices have risen to record levels
lately.
    "The market is just waiting for more new government bond
issues," he said.'''



#print body

#top=['grain','wheat','corn','oat','rye','sorghum','soybean','oilseed']

X=np.matrix(text_normalize(body))
X=X.reshape(X.size,1)


topics=pd.read_csv('all-topics-strings.lc.txt',)
topics_vec=np.matrix(topics)

Y_test=np.zeros((1,topics_vec.shape[0]))
'''for x in range(0,topics_vec.shape[0]):
    if topics_vec[x,0] in top:
        Y_test[0,x]=1'''
#print Y_test.shape
#exit()


vocabdic={}

file=open('Vocab_dir.csv','r')
data=file.readline().replace('\r\n','')
data=data.split(',')
print len(data)
print data

for x in range(0,len(data)):
    vocabdic.update({data[x]:x})
#print vocabdic
print len(vocabdic)
wind=[]
for x in range(0,X.shape[0]):
    if vocabdic.get(X[x,0])!=None:
        wind+=[vocabdic.get(X[x,0])]
print 'Out of %d words %d found in vocab!' % (X.shape[0],len(wind))
file.close()


X_test=np.zeros((len(vocabdic),1))
#print wind
if len(wind)>0:
    X_test[np.matrix(wind),0]=1


X_test=X_test.reshape(1,X_test.shape[0])
print X_test.shape


data=scio.loadmat('X_train.mat')
X_train=np.matrix(data['X_train'])
data=scio.loadmat('Y_train.mat')
Y_train=np.matrix(data['Y_train'])

LSVM= OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(X_train, Y_train )

pred=LSVM.predict(X_test)
#print Y_test
print pred.reshape((1,pred[0,:].size))

pos=np.where(pred[0,:]==1)[0]
print 'The topics Classification for the above document is-'
for x in pos:
    print topics_vec[x,0]
print pos
count=0
#print len(np.where(Y_test[0,:]==pred[0,:].reshape((1,pred[0,:].size))))


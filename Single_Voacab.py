__author__ = 'Karan Chopra'

'''
#Use the vocabs earlier created
#to produce a single vocab directory
#to use it as the feature vector for
#training and testing of the documents.
'''

import pandas as pd
import numpy as np
import csv

vocablist=None

topics=pd.read_csv('all-topics-strings.lc.txt',)
topics_vec=np.matrix(topics)

for x in range(topics_vec.shape[0]):
        topics_vec[x,0]=topics_vec[x,0].strip()
#print topics_vec

for top in topics_vec[:,0]:
        vocabdir=open('Vocab'+top[0,0]+'.csv','r')
        vocab=vocabdir.readline().strip().replace('\r\n','')
        vocab=vocab.split(',')[1:]
        #print vocab
        if len(vocab)<1:
            continue
        if vocablist==None:
            vocablist=vocab
        else:
            for w in vocab:
                if w not in vocablist:
                    vocablist+=[w]
        vocabdir.close()
#print vocablist
vo=open('Vocab_dir.csv','w')
writer=csv.writer(vo)
writer.writerow(vocablist)
vo.close()
print 'Created the Single Vocab Directory!'
print 'The feature vector size-' +str(len(vocablist))
        #writer=csv.writer(vocabdir)
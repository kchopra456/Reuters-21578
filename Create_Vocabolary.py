__author__ = 'Karan Chopra'
'''
#To generate individual vocabs for the labels of the Topics tag and also segregate the Useful documents
'''

import re
import string
from stemming.porter2 import stem
import numpy as np
import pandas as pd
import csv

vocabdic={}
''' ### Retrieve the topics tags ### '''
topics=pd.read_csv('all-topics-strings.lc.txt',)
topics_vec=np.matrix(topics)
topics=[]
doc_id=-1

''' ### Function to retieve the Topic List from the given data set ### '''

def get_topicat(line):
    line=line.strip()
    line=line.replace('<TOPICS>','<D>').replace('</TOPICS>','<D>').replace('</D>','<D>')
    topics=line.split('<D>')
    topics=filter(None,topics)
    return topics


def update_list(wlist,vocab):
    for word in vocab:
        #print word
        rlist=[]
        if wlist==None:
            for x in range(0,len(vocab)):
                if vocab[x] not in vocab[0:x] and vocab[x] not in range(x+1,len(vocab)):
                    rlist+=[vocab[x]]
            return rlist

        if word in wlist:
            continue
        else:
            wlist+=[word]
    return wlist

''' ### Function to allow union of the current label vocab with the normalized body data ### '''

def add_to_dict(topics,vocab):
    global vocabdic
    for top in topics:
        #print top
        #print np.where(topics_vec==top)
        index=np.where(topics_vec==top)
        wlist=vocabdic.get(index[0][0])
        #print wlist
        wlist=update_list(wlist,vocab)
        vocabdic.update({str(index[0][0]):wlist})
        #print vocabdic
        #print wlist

''' ### Function that generates the normalized word list from the body tag data ### '''

def text_normalize(line):
    words=[]


    ''' ### Convert to the lower case email contents### '''
    line=string.lower(line)
    #print(line)

    ''' ### Strip the HTML tags ### '''
    regex=re.compile("[<\[^<>\]+>]")
    line=regex.sub('',line)
    #print(line)

    ''' ### Process Numbers ### '''
    regex=re.compile('[0-9]+')
    line=regex.sub('number',line)
    #print(line)

    ''' ### Process URL ### '''
    regex=re.compile('(http|https)://[\S]*')
    line=regex.sub('httpaddr',line)
    #print line

    ''' ### Process Email Address ### '''
    regex=re.compile('[\S]+@[\S]+')
    line=regex.sub('emailaddr',line)
    #print(line)

    ''' ### Process Dollar Sign ### '''
    line=re.sub('[$]+','dollar',line)
    #print(line)

    ''' ### remove Puntuaution ### '''
    line=line.translate(None,string.punctuation)
    #print( line)

    ''' ### TOkenize the list ### '''
    words+=line.split(' ')
    #print words

    ''' ### Remove Non alpha Numeric ### '''
    words=map(lambda x:re.sub('[^a-zA-z0-9]','',x),words)
    #print(words)
    words=filter(None,words)

    ''' ### Stem the Strings ### '''
    words=map(lambda x:stem(x),words)

    return words

''' ### Function to save the classified documents with NEWID in the file-name for further traing data ### '''

def save_document(vocab,topics,doc_id):
    #print doc_id
    file=open('.\documents1\doc_'+str(doc_id).strip()+'.csv','w')
    writer=csv.writer(file)
    writer.writerow(vocab)
    Y=np.zeros((topics_vec.shape[0],1))
    Y[np.where(topics_vec==topics)[0]]=1
    Y=Y.flatten()
    writer.writerow(Y)
    file.close()

''' ### This function parses A SGML file and extract XML tags data and processes the body further ### '''

def read_file(file):
    global topics
    global doc_id
    for x in range(0,len(file)):
        line=file[x]
        #print file[x]

        if line.startswith('<REUTERS'):

            doc_id=line.split('NEWID="')[1].replace('">','')

        if line.startswith('<TOPICS>'):
            topics=get_topicat(line)
        body=''
        #print topics
        if len(topics)==0:
            continue

        if line.__contains__('<BODY>'):
            while True ^ line.__contains__('</BODY>'):
                body+=line
                x+=1
                line=file[x]
            body=body.split('<BODY>')[1].replace('Reuter','')
            vocab= text_normalize(body)
            #print vocab
            add_to_dict(topics,vocab)
            #print(doc_id)
            save_document(vocab,topics,doc_id)
            topics=[]





def main():

    for x in range(topics_vec.shape[0]):
        topics_vec[x,0]=topics_vec[x,0].strip()
    #print(topics_vec)
    ''' ### Extract the data from multiple SGML FILES ### '''
    files=['reut2-000.sgm','reut2-001.sgm','reut2-002.sgm','reut2-003.sgm','reut2-004.sgm','reut2-005.sgm','reut2-006.sgm','reut2-007.sgm','reut2-008.sgm','reut2-009.sgm','reut2-010.sgm','reut2-011.sgm','reut2-012.sgm','reut2-013.sgm','reut2-014.sgm','reut2-015.sgm','reut2-016.sgm','reut2-017.sgm','reut2-018.sgm','reut2-019.sgm','reut2-020.sgm','reut2-021.sgm']
    for filename in files:
        filereut=open(filename,'r')
        file=filereut.readlines()
        read_file(file)
        filereut.close()
        print 'Completed Parsing-' + filename


    for top in topics_vec[:,0]:
        vocabdir=open('Vocab'+top[0,0]+'.csv','w')
        writer=csv.writer(vocabdir)
        if vocabdic.get(str(np.where(topics_vec==top)[0][0]))!=None:
            #print top[0,0], len(vocabdic.get(str(np.where(topics_vec==top)[0][0]))), vocabdic.get(str(np.where(topics_vec==top)[0][0]))
            writer.writerow([top[0,0]]+vocabdic.get(str(np.where(topics_vec==top)[0][0])))
        else:
            #print top[0,0]
            writer.writerow([top[0,0]])
        vocabdir.close()
    print '##The labels with no corresponding documents have no Vocab build, thus there will be no document classified as one of them!##'
    print 'Vocabulory Building Complete!- Use Single_vocab.py to create Vocab Directory'



if __name__=='__main__':main()


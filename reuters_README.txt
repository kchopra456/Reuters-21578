# REUTERS data set multiclass classifcation

1. Approach

In order to classify the document , which are not categorized under any category; we need to learn from the documents that are currently labeled and train our parameters in a way that we can classify others using those parameters. This means we extract information (text extraction here) from these documents and use their actual labels as a means for the accuracy check.
Thus I am using the SVM classification approach as One-vs-All classifier to train my parameters.

2. Solution

We require to classify the data set with multiple labels (135) under the category ‘topics’. Thus One-vs-All technique with the SVM (Linear Kernel) text classification was used. The main aim was to design a vocabulary of words that occur in the documents that are currently labeled and use this vocab dir as a feature vector. The Vocab Dir comes out to be a list of 3000+ words, many of the words may not even act as a feature such as (‘this’ , ‘a’), we can use tf-idf (term frequency- inverse document frequency). This would reduce the feature vector size or another approach can be using only those words in the vocab for a label which occur in at least 10% of the documents for that label.
I used- OneVsRestClassifier from the sklearn.multiclass library to classify the documents into multiple labels. So for each document a prediction vector was computed of size 135 where if index (i)= 1 then that document belong to the i th label of topics.
After the training of the data set, any document can be used with the Un-Classified classification script.
It will list all the labels the document belong to.


3. Process

The documents in the data set are dumped with XML tags in a SGML file format. Where the documents are either categorized with a label under ‘topics’ or not. The documents that are not categorized are of no current use, under the classification approach.
Each SGML file was parsed and with the tag identification of the useful documents, the ‘body’ tag data was normalized (punctuation removal and stemming + processing) and a vocabulary was generated for the 135 labels under the ‘topics’ category by union of the vocab list. Each of these documents was also separately stored in the ‘.\documents\’ directory for the training and testing phase.
(Use Create_Vocabulory.py)

Then we generate a vocab directory that is the union of all the vocabs. This directory works as our feature vector. While determining a label for a simple body tag data we create a vector of size equal to the words in the vocab dir and mark the words indices which occur in the body tag. Thus for the words in the body after normalization are looked up in the word index dictionary and corresponding indices are marked 1.
(Use Single_Vocab.py)

After I have created the Vocab Dir, I use the saved labeled documents and create a feature matrix with each document as a row in this matrix and the columns are the feature vectors corresponding to the vocab dir. The features matrix is split into train and test set with the split ration of 70/30. Stored as X(Y)_train.mat and X(Y)_test.mat. So I do not have to parse each time from such a large data set and I can directly get the features from these files. 
(Use Train.py)

Then finally I train the parameters by loading the .mat files and use OnevsRest SVM classifier to train on my data set that is 70 % of the actual feature matrix (shuffled in order to introduce randomness in the data ) and check my training accuracy on the rest of the 30% data set.
My training accuracy comes out to be 99.6% on the test set (the test set was not included in the training set)!
(Use SVM_Classification.py)

Then to finally create a classifier for unlabeled data we can directly modify the script to add the body tag data to the body variable and this script will normalize this data as others were done and create a feature matrix for this data, and gain train the parameters and then predict the class for this document body tag data.
The script lists all the labels for the document on the terminal.
(Use Un_Classified_Classification.py)

Note- WE can modify the last script to parse the SGML files and extract the unlabeled data and save them with label.


4. Results

99.6 % accuracy on the labeled test data.
Can directly use 'Un_Classified_Classification.py' to classify any docuemnt uneder the topics label.

Classification can be impoved by using tf-idf.

5. Code Files
(In Execution Sequence)
Create_Vocabulory.py
Single_Vocab.py
Train.py
SVM_Classification.py
Un_Classified_Classification.py

6. Output Files

Not such any file, the labels are displayed on the terminal by using the Un_Classified_Classification.py

from __future__ import print_function
import numpy as np
import pandas as pd
import torch as t
from sklearn.model_selection import train_test_split
from ConvClassifier import Brain

#Hyperparameters
###################################
kernel_height = [4,5,6]
out_channels = 40
dropout_prob= 0.3
embedding_dim = 80
# hidden_dim = 100
output_size = 2
epochs = 30
###################################

#File Loading
###################################
filename = 'E:\AI_files\Sentiment_analysis\data_mod_filtered.csv'
df = pd.read_csv(filename,encoding = "ISO-8859-1")
###################################

#Removing punctuations from the text
###################################
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

for i in range(len(df.iloc[:,1])):
        no_punct = ""
        for char in df.iloc[i,1]:
            if char not in punctuations:
                no_punct = no_punct + char
        df.iloc[i,1] = no_punct
###################################

#Creating Vocab from the corpus of data
###################################
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return t.tensor(idxs, dtype=t.long)

word_to_ix = {}
for i in range(len(df.iloc[:,1])):
    for word in (df.iloc[i,1].split()):
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
###################################
vocab_size = len(word_to_ix)

#Initializing and creating the Ai model
nn_Brain = Brain(kernel_height,out_channels,dropout_prob,embedding_dim, vocab_size,output_size)


#seggrating, creating and shuffling the sample and test data and its corresponding labels
###################################
samp_train, samp_test, label_train, label_test = train_test_split(df.iloc[:,1], df.iloc[:,0], test_size= 0.2, random_state=43)
samp_train = samp_train.values
###################################

#Train the model
nn_Brain.pre_trainer(samp_train,label_train,word_to_ix,epochs)


#Testing the model
###################################
samp_test = samp_test.values
label_test = label_test.values
n=0
for i in range(len(samp_test)):
    sentence = prepare_sequence(samp_test[i].split(),word_to_ix)
    out_data = nn_Brain.classify(sentence)
    out_data = out_data.numpy()
    print(out_data)
    print(label_test[i])
    if out_data == label_test[i]:
        n+=1
accuracy = n / len(samp_test) *100
print(accuracy)
###################################
#print(label_train.shape)
#print(labels_class)







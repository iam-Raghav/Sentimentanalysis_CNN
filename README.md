# Sentimentanalysis_CNN
Sentiment Analysis done using CNN (Convolutional Neural Net)

# Pre-requistics
    Python- 3.7
    Torch - 1.0.0
    numpy -1.16.4
    sklearn - 0.21.3
    nltk - 3.4.4
    pandas - 0.25.0
    
# Files
1. data.csv - Contains the dataset having the sentence and its sentiment label.
2. Text_cleaner.py - This file will clean the dataset by removing the stopwords, punctuations and converting the format to UTF-8
3. Conv_preprocessing.py - This file contains the main code that needs to be run to predict the sentiment.
4. ConvClassifier.py - This file contains the training, testing and building a CNN model required for the sentiment analysis.

Note:- Text cleaner file is made as a separate file for re-usability and saving time to avoid running the text cleaning process for each and every time Pre_processing.py file is run.
    
# Downloads and Setup
Once you clone this repo, run the Conv_preprocessing.py file to do the sentiment analysis and to train the model.

# Evalution metric
Evalution metric used here is the accuracy.

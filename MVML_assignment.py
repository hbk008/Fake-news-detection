
"""
Created on Wed Aug 18

@author: Harshit Bokadia

MVML assignment
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re 
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Stopwords are the words which occur very frequently in a corpus 
# and are removed for normalizing the text.

# Downloading the stopwords.
nltk.download('stopwords')
print(stopwords.words('english'))

# Load training data.
train_data = pd.read_csv(r'C:\Users\harsh\Desktop\HBK008\Projects\MVML\data\train.csv')
# Load test data.
test_data = pd.read_csv(r'C:\Users\harsh\Desktop\HBK008\Projects\MVML\data\test.csv')
# Load test data labels.
test_labels = pd.read_csv(r'C:\Users\harsh\Desktop\HBK008\Projects\MVML\data\labels.csv')

# Check the shape and column id's of training data.
print("Training data shape:", train_data.shape)
print("Training data column id's", train_data.columns)

# Look at the first 10 data points.
train_data.head(10)

# Distribution of dataset in both the classes.
print(train_data.label.value_counts());

# Check missing values in the dataset.
train_data.isnull().sum()

# Replace the null values with empty strings. 
train_data = train_data.fillna('')

## Data to be used for prediction:
# I am using 'author' and 'title' data columns to make prediction. 
# Including 'text' takes a lot of time for stemming. 
# Since the assigment is to be done in 1-3 hours, I am skippping 'text'. 
# If we want to include 'text', we can do so using line 58 instead of 59.

#train_data['content'] = train_data['author'] + ' ' + train_data['title'] + ' : ' + train_data['text']
train_data['content'] = train_data['author'] + ' ' + train_data['title'] 
print(train_data['content'])

# Defining the data preprocessing function below.

ps = PorterStemmer()

def data_preprocessing(data):
    # Pick all lowercase and uppercase characters.
    # Numbers/punctuations will be removed and replaced by a whitespace.
    processed_data = re.sub('[^a-zA-Z]', ' ', data)
    processed_data = processed_data.lower().split()
    # Applying stemming (to reduce the word to it's root word) and remove stopwords.
    processed_data = [ps.stem(word) for word in processed_data if not word in stopwords.words('english')]
    processed_data  = ' '.join(processed_data )
    return processed_data

# Preprocessing the training data.
train_data['content'] = train_data['content'].apply(data_preprocessing)
print(train_data['content'])

# Defining X_train and Y_train from the training data. 
X_train = train_data['content'].values
Y_train = train_data['label'].values

# Vectorizing the training data to convert into numerical format. 
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)

# Building logistic regression classifier.
model = LogisticRegression()
model.fit(X_train, Y_train)

## Model evaluation
# We evaluate the model by the accuracy it achieves on the test data. The test data accuracy
# is very low (~64%) compared to training data which has 98.83% accuracy.
# This shows model generalizes poorly.
# # We can improve the model by including the text column in the data which 
# I skipped due to time constraint. 
# We can try other models like LSTM and transformers as well.

# train data evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = metrics.accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of training data :', training_data_accuracy)

# test data preprocessing and evaluation. 

# Check for missing values in the dataset.
test_data.isnull().sum()

# Replace them with emtpy strings. 
test_data = test_data.fillna('')

# Using only 'author' and 'title' columns same as training data.
# test_data['content'] = test_data['author'] + ' ' + test_data['title'] + ':' + test_data['text']
test_data['content'] = test_data['author'] + ' ' + test_data['title']

# Preprocessing the test data.
test_data['content'] = test_data['content'].apply(data_preprocessing)

# Define X_test and Y_test [test_data and test_labels are assigned from test.csv and labels.csv].
X_test = test_data['content'].values
Y_test = test_labels['label'].values

# Vectorize the test data [DO NOT fit again as the dimensions need to be same for training data and test data].
X_test = vectorizer.transform(X_test)

# Predictions on test data from logistic regression model trained on training data.
X_test_prediction = model.predict(X_test)
testing_data_accuracy = metrics.accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of testing data :', testing_data_accuracy)

# Confusion matrix
cm = metrics.confusion_matrix(Y_test, X_test_prediction)
print(cm)

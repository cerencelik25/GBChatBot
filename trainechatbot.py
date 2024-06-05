# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:44:02 2024

@author: ceren
"""

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
import random
import tensorflow as tf
print(tf.__version__)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ['?', '!', '@', '$']

# Use json to load data
with open('intents.json') as data_file:
    intents = json.load(data_file)

# Print the contents of the JSON file to understand its structure
print(intents)

# Populating the lists
for intent in intents['intents']:
    # Print each intent to see its structure
    print(intent)
    for pattern in intent['patterns']:
        
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        print("w in for loop : - ",w)
        words.extend(w)
        
        # Add documents
        documents.append((w, intent['tag']))
        
        # Add classes to our class list
        print("classes : -", classes)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "Documents:", documents)
print(len(classes), "Classes:", classes)
print(len(words), "Unique lemmatized words:", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#initializing the training data
training = []
output_empty = [0]*len(classes)

for doc in documents:
    #initializing the bag of word
    bag = []
    #list of tokenized words for the pattern
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]    
    
    for w in words:
        bag .append(1) if w in pattern_words else bag.append(0)
        
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag,output_row])

print("training : ", training)
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")
print("train_x: ", train_x)
print("train_y: ", train_y)

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate= 0.01, decay=1e-6, momentum=0.9, nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=35, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('model created')



    
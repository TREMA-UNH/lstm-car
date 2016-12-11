#!/usr/bin/python

import numpy as np
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Activation
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

steps = 5

embedding = {}
for line in open('glove.6B.100d'):
    xs = line.split()
    word = line[0]
    vec = np.array(map(float, xs[1:]))
    embedding[word] = vec

tokenizer = Tokenizer()
text = [ ' '.join(w for w in line.split()[:steps+1]) for line in open('sentences.txt') ]
tokenizer.fit_on_texts(text)
lines = tokenizer.texts_to_sequences(text)
vocab_size = len(tokenizer.word_index)

lines = [line for line in lines if len(line) > steps+1]

train_x = np.zeros([len(lines), steps, vocab_size])
train_y = np.zeros([len(lines), steps, vocab_size])
for i, line in enumerate(line for line in lines):
    for j, word in enumerate(line[:steps]):
        train_x[i, j, word] = 1
        train_y[i, j, line[j+1]] = 1

model = Sequential()
print vocab_size
print 'a'
model.add(LSTM(vocab_size, return_sequences=True, input_dim=vocab_size, input_length=steps))
model.add(Activation('softmax'))
print 'b'

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy'
              )

print 'fitting'
model.fit(train_x, train_y)

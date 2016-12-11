#!/usr/bin/python

import collections
import numpy as np
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Activation, Dense, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

steps = 30

embeddings = {}
for line in open('glove.6B.50d.txt'):
    xs = line.split()
    word = xs[0]
    vec = np.array(map(float, xs[1:]))
    embeddings[word] = vec

dim = len(embeddings['the'])

tokenizer = Tokenizer()
lines = [ line.split() for line in open('articles')]
print set(w for w in line for line in lines if w not in embeddings)
text = [ [ embeddings[w] for w in line if w in embeddings ]
         for line in lines ]
text = [np.vstack(line[:steps+1]) for line in text if len(line) > steps+1]
text = np.dstack(text).swapaxes(1,2).swapaxes(0,1)
print text.shape
train_x = text[:,:-1,:]
train_y = text[:,1:,:]

model = Sequential()
print 'a'
model.add(LSTM(dim, return_sequences=True, input_dim=dim, input_length=steps))
model.add(Dense(dim))
model.add(Activation('tanh'))
print 'b'

model.compile(optimizer='rmsprop',
              loss='mse'
              )

print 'fitting'
model.fit(train_x, train_y)

with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('weights.hdf')

print 'preparing word index',
class WordIndex(object):
    def __init__(self, words):
        idx = collections.defaultdict(lambda: [])
        for k,v in embeddings.items():
            idx[tuple(x > 0 for x in v[:10])].append((k,v))
        print 'done'
        self.word_index = idx

    def get(self, vec):
        word = None
        sim = None
        key = tuple(x > 0 for x in vec[:10])
        for k,v in self.word_index.get(key, {}):
            tmp = np.dot(v, vec) / np.linalg.norm(vec) / np.linalg.norm(v)
            if tmp > sim:
                word = k
                sim = tmp
        return word

idx = WordIndex(embeddings)

print idx.get(embeddings['the'])
for line in model.predict(train_x):
    print ' '.join(idx.get(em) for em in line)

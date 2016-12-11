#!/usr/bin/env python3

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

from trec_car import read_data
from utils import *

maxlen = 40
step = 3

def find_nth_word(str, n):
    i = 0
    for iter in range(0,n):
        i = str.find(' ', i+1)
    return i

paras = [ ' '.join(words) for words in read_paras() ]
#text = open('articles').read().lower()
vocab = sorted(list(set(c
                        for text in paras
                        for c in text)))
print(vocab)
vocab_indices = dict((c, i) for i, c in enumerate(vocab))
indices_vocab = dict((i, c) for i, c in enumerate(vocab))

# Compute training samples
input_sequences = []
next_elems = []
for line in paras[:10000]:
    for i in range(0, len(line) - maxlen, step):
        input_sequences.append(line[i: i + maxlen])
        next_elems.append(line[i + maxlen])

comparable_input_sequences = []
next_chunks = []
for line in paras[:10000]:
    cut_offset = find_nth_word(line, maxlen)
    cut_nextword = find_nth_word(line, maxlen+1)
    comparable_input_sequences.append(line[cut_offset-1-maxlen:cut_offset-1])
    next_chunks.append(line[cut_offset:cut_nextword])


print('nb sequences:', len(input_sequences))

# Vectorize training samples
print('Vectorization...')
train_x = np.zeros((len(input_sequences), maxlen, len(vocab)), dtype=np.bool)
train_y = np.zeros((len(input_sequences), len(vocab)), dtype=np.bool)
for i, sequence in enumerate(input_sequences):
    for t, elem in enumerate(sequence):
        train_x[i, t, vocab_indices[elem]] = 1
    train_y[i, vocab_indices[next_elems[i]]] = 1

# Construct model
model = Sequential()
model.add(LSTM(128, input_dim=len(vocab), input_length=maxlen))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

text = ' '.join(paras)
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(train_x, train_y, batch_size=128, nb_epoch=1)

    prefixlen = maxlen
    def ld_test_predict(model):
        for row in range(0,maxlines,100):

            seed = train_x[row, 0:prefixlen]
            generated = seed[:]

            x = np.zeros((1, maxlen, len(vocab)))
            for t, elem in enumerate(generated[-maxlen:]):
                x[0, t, vocab_indices[elem]] = 1.
            preds = model.predict(x,verbose=0)[0]
            next_elem_idx = sample(preds, diversity)
            next_elem = indices_vocab[next_elem_idx]
            generated += next_elem

        print(seed,'\t',generated[prefixlen:])
        print()


    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)
        for start_index in range(0, train_y.shape[0], 100):
            seed = text[start_index : start_index + maxlen]
            generated = seed[:]

            for i in range(10):
                x = np.zeros((1, maxlen, len(vocab)))
                for t, elem in enumerate(generated[-maxlen:]):
                    x[0, t, vocab_indices[elem]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)  # arent we overwriting our ground truth? (L:D)
                #next_index = np.argmax(preds)
                generated += indices_vocab[next_index]

            print(seed, '\t', generated[maxlen:])
            print()

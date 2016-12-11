#!/usr/bin/python3

import numpy as np
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

from trec_car import read_data
from utils import *

maxlen = 40
step = 3

embeddings = read_glove('glove.6B.50d.txt')
# Word embedding dimension
dim = len(embeddings['the'])

paras = read_paras()

# words unknown to embedding
print(set(w
          for line in paras
          for w in line
          if w not in embeddings))



# Compute training samples
train_x = []
train_y = []
for line in paras[:10000]:
    embedded = [ embeddings[elem]
                 for elem in line
                 if elem in embeddings ]
    print(' '.join( elem if elem in embeddings else '<%s>' % elem
                    for elem in line ))
    for i in range(0, len(embedded) - maxlen - 1, step):
        train_x.append(embedded[i:i+maxlen])
        train_y.append(embedded[i+maxlen+1])

train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x.shape, train_y.shape)

# Construct model
model = Sequential()
model.add(LSTM(128, input_dim=dim, input_length=maxlen))
model.add(Dense(dim))
model.add(Activation('linear'))

optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='cosine_proximity')

# Prune embeddings
seen_words = set(w for w in line for line in paras)
seen_embeddings = { w: v
                    for w,v in embeddings.items()
                    if w in seen_words }
seen_embeddings = embeddings
idx = BinnedEmbeddingIndex(seen_embeddings)

print(idx.get(embeddings['the']))

prefixlen = maxlen
def ld_test_predict(model):
    for row in range(0,maxlines,100):
        seed = train_x[row,0:prefixlen]
        generated = seed[:]

        x = np.zeros((1,maxlen,dim))
        for t, elem in enumerate(generated[-maxlen:]):
            x[0, t, :] = embeddings[elem]
        preds = model.predict(x,verbose=0)[0]
        next_elem_vec = preds
        next_elem = idx.get(next_elem_vec)
        generated += next_elem

    print(seed,'\t',generated[prefixlen:])
    print()


def test_predict(model):
    for idx in range(0, train_x.shape[0], 100):
        seed = train_x[idx, :]
        generated = seed[:]

        for i in range(10):
            x = generated[newaxis, -maxlen:, ]
            for t, elem in enumerate(generated[-maxlen:]):
                x[0, t, vocab_indices[elem]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = str(idx.get(y))
            generated += indices_vocab[next_index]

        print(seed, '\t', generated[maxlen:])

    for x,y in list(zip(train_x, model.predict(train_x)))[::100]:
        print(' '.join(str(idx.get(w)) for w in x[-5:]),'\t', str(idx.get(y)))

for iteration in range(1, 60):
    print('fitting')
    model.fit(train_x, train_y, nb_epoch=1)
    test_predict(model)

with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('weights.hdf')

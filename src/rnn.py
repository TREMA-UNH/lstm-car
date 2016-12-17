#!/usr/bin/python3

import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

from trec_car import read_data
from utils import *


#download_nlk_resources()

maxlen = 40
step = 3
prefixlen = maxlen # prefixlen must be >= maxlen!


embeddings = read_glove('data/glove.6B.50d.txt')
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
test_seq = []
for line in paras[:10000]:
    embedded = [ embeddings[elem]
                 for elem in line
                 if elem in embeddings ]
    embedded = np.array(embedded)
    print(' '.join( elem if elem in embeddings else '<%s>' % elem
                    for elem in line ))
    for i in range(0, len(embedded) - maxlen - 1, step):
        train_x.append(embedded[i:i+maxlen])
        train_y.append(embedded[i+maxlen])
    if len(embedded) > prefixlen+2:
        test_seq.append(embedded)

train_x = np.array(train_x)
train_y = np.array(train_y)

# Construct model
model = Sequential()
model.add(LSTM(128, input_dim=dim, input_length=maxlen))
model.add(Dense(dim))
model.add(Activation('linear'))

optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='cosine_proximity')

# Prune embeddings
# seen_words = set(w for w in line for line in paras)
# seen_embeddings = { w: v
#                    for w,v in embeddings.items()
#                    if w in seen_words }
# idx = BinnedEmbeddingIndex(seen_embeddings)

idx = BinnedEmbeddingIndex(embeddings)

print(idx.get(embeddings['the']))


def wordvecs_to_text(generated):
    return " ".join(idx.get(elem) for elem in generated)


monitor=[]

maxlines=len(test_seq) -1
def ld_test_predict(model):
    for row in range(1,maxlines-1,10):
        seed = test_seq[row][0:prefixlen]
        generated = list(seed[:])

        x = np.zeros((1,maxlen,dim))
        for t, elem in enumerate(generated[-maxlen:]):
            x[0, t, :] = elem
        preds = model.predict(x,verbose=0)[0]
        next_elem_vec = preds
        generated.append(next_elem_vec)

        print("output: ", wordvecs_to_text(seed),'\t ====== \t',wordvecs_to_text(generated[prefixlen:]) )
    print()


def ld_test_score(model):
    count_correct = 0
    count_all = 0
    for row in range(1,maxlines-1,10):
        seed = test_seq[row][0:prefixlen]
        print(seed.__class__)
        candidate1 = test_seq[row-1][prefixlen+1:prefixlen+2]
        candidate2 = test_seq[row][  prefixlen+1:prefixlen+2]
        candidate3 = test_seq[row+1][prefixlen+1:prefixlen+2]


        x = np.zeros((1,maxlen,dim))
        x[:,:,:]=seed
        y1 = np.zeros((1,dim))
        y1[:,:] = candidate1
        y2 = np.zeros((1,dim))
        y2[:,:] = candidate2
        y3 = np.zeros((1,dim))
        y3[:,:] = candidate3

        cands = [y1,y2,y3]

        scored_items = [(cand, model.evaluate(x,cand))  for cand in cands]
        ranking = sorted(scored_items, key=lambda x: -x[1])


        for cand, score in ranking:
            print("score: ", wordvecs_to_text(seed), '\t=========\t', wordvecs_to_text(cand), score)
        is_correct = np.alltrue(ranking[0][0] == candidate2)
        print('correct? ',is_correct)
        if is_correct:
            count_correct += 1
        count_all += 1

    monitor.append(count_correct/count_all)
    print('monitor=',monitor)



# def test_predict(model):
#     for idx in range(0, train_x.shape[0], 100):
#         seed = train_x[idx, :]
#         generated = seed[:]
#
#         for i in range(10):
#             x = generated[np.newaxis, -maxlen:, ]
#             for t, elem in enumerate(generated[-maxlen:]):
#                 x[0, t, vocab_indices[elem]] = 1.
#
#             preds = model.predict(x, verbose=0)[0]
#             next_index = str(idx.get(y))
#             generated += indices_vocab[next_index]
#
#         print(seed, '\t', generated[maxlen:])
#
#     for x,y in list(zip(train_x, model.predict(train_x)))[::100]:
#         print(' '.join(str(idx.get(w)) for w in x[-5:]),'\t', str(idx.get(y)))


for iteration in range(1, 60):
    print('fitting')
    model.fit(train_x, train_y, nb_epoch=20, validation_split=0.5)
    ld_test_predict(model)
    ld_test_score(model)
    print('\n\n')
    #
    # for input in inputs:
    #     scored_items = {candidate:score(model, input, candidate) for candidate in outputs}
    #     ranking = sorted(scored_items.items(), key=lambda k,v: -v)
    #     for cand, score in ranking[0:3]:
    #         print(input,' ',score,' ',cand)


with open('model.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('weights.hdf')

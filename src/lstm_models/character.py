import itertools
from typing import Iterable

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint


from utils import *
from lstm_models import ParaCompletionModel

class CharacterLSTMModel(ParaCompletionModel):
    'LSTM Model with characters  as inputs/outputs'

    def __init__(self, maxlen: int, epochs: int):
        super().__init__()
        self.epochs = epochs
        self.maxlen = maxlen
        self.vocab = ' abcdefghijklmnopqrstuvwxyz'
        self.dict = {v:i for i,v in enumerate(self.vocab)}
        self.ONES = np.ones(len(self.vocab))

        # Construct model
        model = Sequential()
        model.add(LSTM(128, input_dim=len(self.vocab), input_length=maxlen))
        model.add(Dense(len(self.vocab)))
        model.add(Activation('softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        self.model = model

    def load_weights(self, fname: str):
        self.model.load_weights(fname)

    def onehot(self, c):
        arr = np.zeros(len(self.vocab))
        arr[self.dict[c]]=1.0
        return arr

    def decode(self, vec:np.ndarray):
        pred_char_idx = np.argmax(vec)
        return self.vocab[pred_char_idx]

    def unknown_wordvec(self):
        return self.ONES

    def create_seq_vector(self, char_query:List[Word]):
        char_query = [self.onehot(c) for c in ' '.join(char_query) if c in self.dict]
        char_query = np.array(char_query)  # type: np.ndarray
        return char_query

    def _preproc_train(self, training_seqs: Iterable[List[Word]], step: int):
        train_x = []
        train_y = []
        for line in training_seqs:
            char_line = self.create_seq_vector(line)

            # create training prefix-suffix pairs by shifting a window through the sequence
            for i in range(0, len(char_line) - self.maxlen - 1, step):
                train_x.append(char_line[i:i+self.maxlen])
                train_y.append(char_line[i+self.maxlen])
                #  todo padding for sequences that are shorter than maxlen

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return (train_x, train_y)


    def _preproc_train_qa(self, training_pairs: Iterable[Tuple[List[Word],List[Word]]], step: int):
        train_x = []
        train_y = []
        for query,answer in training_pairs:
            char_query = self.create_seq_vector(query)
            char_answer = self.create_seq_vector(answer)


            if len(char_query)>=self.maxlen :
                char_line = np.vstack([char_query[-self.maxlen:], char_answer])
            else:
                zero_entries = self.maxlen - len(char_query)
                padding = np.vstack([self.unknown_wordvec()]*zero_entries)
                char_line = np.vstack([padding, char_query, char_answer])

                # create training prefix-suffix pairs by shifting a window through the sequence
            for i in range(0, len(char_line) - self.maxlen - 1, step):
                train_x.append(char_line[i:i+self.maxlen])
                train_y.append(char_line[i+self.maxlen])

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return (train_x, train_y)

    def train_qa(self, training_pairs: List[Tuple[List[Word],List[Word]]]):
        (train_x, train_y) = self._preproc_train_qa(training_pairs, step=3)
        with open('model.yaml', 'w') as f: f.write(self.model.to_yaml())
        callbacks = [ModelCheckpoint('weights.hdf')]
        self.model.fit(train_x, train_y,
                       nb_epoch=self.epochs, validation_split=0.2, callbacks=callbacks)


    def train(self, training_seqs: List[List[Word]]):
        (train_x, train_y) = self._preproc_train(training_seqs, step=3)
        with open('model.yaml', 'w') as f: f.write(self.model.to_yaml())
        callbacks = [ModelCheckpoint('weights.hdf')]
        self.model.fit(train_x, train_y,
                       nb_epoch=self.epochs, validation_split=0.2, callbacks=callbacks)

    def generate_word(self, test_inputs: Iterable[List[Word]]) -> List[Word]:
        x_list = [[list(self.onehot(c))
                   for c in (' '.join(input + ['']))
                   if c in self.dict]
                  for input in test_inputs]
        test_x = np.array([seq[-self.maxlen:]
                           for seq in x_list],
                          dtype=float)

        print('test_x.shape= ',test_x.shape)

        for i in range(0, 20):
            preds = self.model.predict(test_x)
            test_x = np.hstack([test_x[:,-(self.maxlen-1):,:], preds[:,np.newaxis,:]])

        pred_words = [([self.decode(pred_vec)
                        for pred_vec in test_x[row,(self.maxlen-20):]])
                      for row in range(0, test_x.shape[0]) ]

        pred_words_cut = [''.join(itertools.takewhile(lambda c: c is not ' ', line))
                          for line in pred_words]
        return pred_words_cut


    def rank_word(self,
                  test_seqs: List[TestSeq]) \
            -> List[Tuple[List[Tuple[Word,float]], Word]]:
        raise NotImplementedError
        # def score(seq: TestSeq) -> (List[Tuple[Word,float]], Word):
        #     test_x_1 = np.array( [self.dict.get(elem)
        #                           for elem in ' '.join(seq.sequence)] )
        #     test_x = np.array([test_x_1])
        #
        #     test_cands = [(cand_char, self.dict.get(cand_char))
        #                   for cand_char in ' '.join(seq.candidates)]
        #
        #     scored_items = [(cand_word, self.model.evaluate(test_x, np.array([cand_wordvec]), verbose=False))
        #                     for cand_word, cand_wordvec in test_cands]
        #     ranking = sorted(scored_items, key=lambda x:x[1])
        #     return (ranking, seq.truth)
        #
        # return list(map(score, test_seqs))

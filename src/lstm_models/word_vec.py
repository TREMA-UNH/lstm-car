from typing import Iterable

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

from embeddings import *
from lstm_models import ParaCompletionModel

class WordVecLSTMModel(ParaCompletionModel):
    'LSTM Model with word vectors as inputs/outputs'

    def _wordvecs_to_text(self, wv_seq):
        return " ".join(self.embeddingIndex.lookup(elem) for elem in wv_seq)


    def __init__(self, embeddingIndex: Embeddings, maxlen: int, epochs:int):
        self.epochs = epochs
        self.embeddingIndex = embeddingIndex

        self.dim = embeddingIndex.dim()
        self.ONES = np.ones(self.dim)
        self.maxlen = maxlen

        # Construct model
        model = Sequential()
        model.add(LSTM(128, input_dim=self.dim, input_length=maxlen))
        model.add(Dense(self.dim))
        model.add(Activation('linear'))
        optimizer = RMSprop(lr=0.01)
        model.compile(optimizer=optimizer, loss='cosine_proximity')
        self.model = model

    def load_weights(self, fname: str):
        self.model.load_weights(fname)

    def create_vector_seq(self, line:Iterable[Word]):
        wordvec_line = [ self.embeddingIndex.get(elem)
                         for elem in line
                         if elem in self.embeddingIndex ]
        wordvec_line = np.array(wordvec_line)  # type: np.ndarray
        return wordvec_line


    def _preproc_train(self, training_seqs: Iterable[List[Word]], step: int):
        train_x = []
        train_y = []
        for line in training_seqs:
            wordvec_line = self.create_vector_seq(line)
            # print(' '.join( elem if elem in self.embeddingIndex else '<%s>' % elem
            #                 for elem in line ))

            # create training prefix-suffix pairs by shifting a window through the sequence
            for i in range(0, len(wordvec_line) - self.maxlen - 1, step):
                train_x.append(wordvec_line[i:i+self.maxlen])
                train_y.append(wordvec_line[i+self.maxlen])
                #  todo padding for sequences that are shorter than maxlen

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return (train_x, train_y)

    def unknown_wordvec(self):
        return self.ONES

    def _preproc_train_qa(self, training_pairs: Iterable[Tuple[List[Word],List[Word]]], step: int):
        train_x = []
        train_y = []
        for query,answer  in training_pairs:
            wordvec_query = self.create_vector_seq(query)
            wordvec_answer = self.create_vector_seq(answer)
            # print(' '.join( elem if elem in self.embeddingIndex else '<%s>' % elem
            #                 for elem in line ))

            if len(wordvec_query)>=self.maxlen :
                wordvec_line = np.vstack([wordvec_query[-self.maxlen:], wordvec_answer])
            else:
                zero_entries = self.maxlen - len(wordvec_query)
                padding = np.vstack([self.unknown_wordvec()]*zero_entries)
                wordvec_line = np.vstack( [padding, wordvec_query, wordvec_answer])

            # create training prefix-suffix pairs by shifting a window through the sequence
            for i in range(0, len(wordvec_line) - self.maxlen - 1, step):
                train_x.append(wordvec_line[i:i+self.maxlen])
                train_y.append(wordvec_line[i+self.maxlen])

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        return (train_x, train_y)


    def train_qa(self, training_pairs: List[Tuple[List[Word],List[Word]]]):
        (train_x, train_y) = self._preproc_train_qa(training_pairs, step=3)
        callbacks = [ModelCheckpoint('weights.hdf')]
        self.model.fit(train_x, train_y,
                       nb_epoch=self.epochs, validation_split=0.2, callbacks=callbacks)

    def train(self, training_seqs: List[List[Word]]):
        (train_x, train_y) = self._preproc_train(training_seqs, step=3)
        callbacks = [ModelCheckpoint('weights.hdf')]
        self.model.fit(train_x, train_y,
                       nb_epoch=self.epochs, validation_split=0.2, callbacks=callbacks)

    def generate_word(self, test_inputs: List[List[Word]]) -> List[Word]:
        test_x = [[self.embeddingIndex.get(elem)
                            for elem in input
                            if elem in self.embeddingIndex]
                           for input in test_inputs]
        def transform_row(test_x_1:list):
            if len(test_x_1) >= self.maxlen:
                test_x_1 = np.ndarray(test_x_1[-self.maxlen:])
            else:
                zero_entries = self.maxlen - len(test_x_1)
                padding = np.vstack([self.unknown_wordvec()]*zero_entries)
                test_x_1 = np.vstack([padding, test_x_1])
            return test_x_1

        test_x = np.array(list(map(transform_row, test_x)))

        preds = self.model.predict(test_x)
        pred_words = [self.embeddingIndex.lookup(pred) for pred in preds]
        return pred_words


    def rank_word(self,
                  test_seqs: List[TestSeq]) \
            -> List[Tuple[List[Tuple[Word,float]], Word]]:

        def score(seq: TestSeq) -> (List[Tuple[Word,float]], Word):
            test_x_1 = np.array( [self.embeddingIndex.get(elem)
                                  for elem in seq.sequence
                                  if elem in self.embeddingIndex] )

            if test_x_1.shape[0] >= self.maxlen:
                test_x_1 = test_x_1[-self.maxlen:]
            else:
                zero_entries = self.maxlen - test_x_1.shape[0]
                padding = np.vstack([self.unknown_wordvec()]*zero_entries)
                test_x_1 = np.vstack([padding, test_x_1])

            test_x = np.array([test_x_1])

            test_cands = [(cand_word, self.embeddingIndex.get(cand_word))
                          for cand_word in seq.candidates
                          if cand_word in self.embeddingIndex]

            scored_items = [(cand_word, self.model.evaluate(test_x, np.array([cand_wordvec]), verbose=False))
                            for cand_word, cand_wordvec in test_cands]
            ranking = sorted(scored_items, key=lambda x:x[1])
            return (ranking, seq.truth)

        return list(map(score, test_seqs))

        # if len(wordvec_query)>=self.maxlen :
        #     wordvec_line = np.vstack([wordvec_query[-self.maxlen:], wordvec_answer])
        # else:
        #     zero_entries = self.maxlen - len(wordvec_query)
        #     padding = np.vstack([self.unknown_wordvec()]*zero_entries)
        #     wordvec_line = np.vstack( [padding, wordvec_query, wordvec_answer])

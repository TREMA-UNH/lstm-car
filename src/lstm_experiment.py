from data_preproc import *
from lstm_models import *



training_seqs = get_training_seqs()
lstmWordvec = WordVecLSTMModel(BinnedEmbeddings('data/glove.6B.50d.txt'), 40)

lstmWordvec.train(training_seqs)

test_seqs = get_test_seqs()

rankings = lstmWordvec.rank_word(test_seqs)

for ranking in rankings[0:3]:
    print(ranking)


input_seqs = [seq.sequence for seq in test_seqs]
next_words = lstmWordvec.generate_word(input_seqs)

predictions = list(zip(test_seqs, next_words))
print(predictions[0:3])


         # generate_word(self, test_inputs: List[List[Word]])




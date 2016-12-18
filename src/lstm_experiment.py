from data_preproc import *
from lstm_models.word_vec import WordVecLSTMModel
from lstm_models.character import CharacterLSTMModel
from embeddings import BinnedEmbeddings
from evaluation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--paragraphs', required=False, default='data/release.paragraphs')
parser.add_argument('-e', '--epochs', required=False, default=1, type=int)
args = parser.parse_args()


training_seqs = get_training_seqs(open(args.paragraphs, 'rb'))
lstmWordvec = WordVecLSTMModel(BinnedEmbeddings('data/glove.6B.50d.txt'), 40, args.epochs)

lstmWordvec.train(training_seqs)

test_seqs = get_test_seqs(open(args.paragraphs, 'rb'))

rankings = lstmWordvec.rank_word(test_seqs)

for ranking in rankings[0:10]:
    print(ranking)

input_seqs = [seq.sequence for seq in test_seqs]
next_words = lstmWordvec.generate_word(input_seqs)

predictions = list(zip(test_seqs, next_words))
for pred, next in predictions[0:10]:
    print(' '.join(pred.sequence), '\t====\t',next)


mrr_score = mrr(rankings)
prec1_score = prec1Rank(rankings)
print("MRR of rankings: ", mrr_score)
print("P@1 of rankings: ", prec1_score)

prec1_gen = prec1Pred(next_words, [seq.truth for seq in test_seqs])
print("P@1 of generations: ", prec1_gen)





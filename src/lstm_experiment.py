from data_preproc_qa import *
from lstm_models import ParaCompletionModel
from lstm_models.word_vec import WordVecLSTMModel
from lstm_models.character import CharacterLSTMModel
from embeddings import BinnedEmbeddings
from evaluation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--paragraphs', required=False, default='data/release.paragraphs')
parser.add_argument('-t', '--testparagraphs', required=False, default='data/release.paragraphs')
parser.add_argument('-e', '--epochs', required=False, default=1, type=int)
parser.add_argument('-l', '--lines', required=False, default=-1, type=int)
args = parser.parse_args()


training_seqs = get_training_seqs(open(args.paragraphs, 'r'), lines=args.lines)  # 'rb' for cbor, 'r' for csv
lstmWordvec = WordVecLSTMModel(BinnedEmbeddings('data/glove.6B.50d.txt'), 40, args.epochs)
lstmChar = CharacterLSTMModel(40, args.epochs)

lstmWordvec.train_qa(training_seqs)
lstmChar.train_qa(training_seqs)


def evaluate(model: ParaCompletionModel, test_seqs: List[TestSeq], generate=True):
    rankings = model.rank_word(test_seqs)

    if generate:
        input_seqs = [seq.sequence for seq in test_seqs]
        next_words = model.generate_word(input_seqs)

        predictions = list(zip(test_seqs, next_words))
        for pred, next in predictions[0:10]:
            print(' '.join(pred.sequence), '\t====\t',next)

        prec1_gen = prec1Pred(next_words, [seq.truth for seq in test_seqs])
        print("P@1 of generations: ", prec1_gen)

    mrr_score = mrr(rankings)
    prec1_score = prec1Rank(rankings)
    print("P@1 of rankings: ", prec1_score)
    print("MRR of rankings: ", mrr_score)


test_seqs = get_test_seqs(open(args.testparagraphs, 'r'), args.lines) # 'rb' for cbor, 'r' for csv
print('word vector:')
evaluate(lstmWordvec, test_seqs)
print('per-character:')
evaluate(lstmChar, test_seqs)

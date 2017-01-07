from data_preproc_qa import *
import lstm_models
from lstm_models import ParaCompletionModel
from lstm_models.word_vec import WordVecLSTMModel
from lstm_models.character import CharacterLSTMModel
from embeddings import BinnedEmbeddings
from evaluation import *
from trec_car.format_runs import *
import argparse

parser = argparse.ArgumentParser()
sub = parser.add_subparsers()

a = sub.add_parser('train')
a.set_defaults(mode='train')
a.add_argument('-m', '--model', type=str, action='append', default=[])
a.add_argument('-E', '--embeddings', type=argparse.FileType('r'), required=False, default='data/glove.6B.50d.txt')
a.add_argument('-p', '--paragraphs', type=argparse.FileType('r'), required=False)
a.add_argument('-L', '--max-length', type=int, help='training sequence length', default=40)
a.add_argument('-e', '--epochs', type=int, required=False, default=1)
a.add_argument('-l', '--lines', type=int, required=False, default=-1)

a = sub.add_parser('test')
a.set_defaults(mode='test')
a.add_argument('-p', '--paragraphs', type=argparse.FileType('r'), required=False)
a.add_argument('-l', '--lines', type=int, required=False, default=-1)
a.add_argument('-m', '--model', type=str, required=True)
args = parser.parse_args()

knownModels = { 'char': CharacterLSTMModel,
                'wordvec': WordVecLSTMModel
                }

def evaluate_next_word(model: ParaCompletionModel, test_seqs: List[TestSeq], generate=True):
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


def evaluate_qa(model:ParaCompletionModel, test_seqs: List[TestSeq], generate=False):
    # test_data = [(seq.query_id, seq.sequence, seq.para_cands) for seq in test_seqs]
    rankings = model.test_qa(test_seqs) # List[Tuple[QueryId, List[Word], List[Tuple[ParagraphId, List[Word]]]]]

    rankingEntries = \
        [
            RankingEntry(query_id, paragraph_id, rank0 + 1, score)
            for (query_id, ranking) in rankings
            for rank0, (paragraph_id, score) in enumerate(ranking)
        ]

    writer = open('outputranking.run', 'w')
    format_run(writer, rankingEntries, model.name())
    writer.close()

if args.mode == 'train':
    models = []   # Type: List[ParaCompletionModel]
    for m in set(args.model):
        mm = knownModels.get(m)
        if mm is CharacterLSTMModel:
            models.append(CharacterLSTMModel(args.max_length))
        elif mm is WordVecLSTMModel:
            embeddings = BinnedEmbeddings(args.embeddings.name)
            models.append(WordVecLSTMModel(embeddings, args.max_length))
        else:
            raise RuntimeError('Unknown model %s' % m)

    training_seqs = get_training_seqs(args.paragraphs, lines=args.lines)
    for m in models:
        m.train_qa(training_seqs, epochs=args.epochs)
        m.save('%s.model' % m.name())

elif args.mode == 'test':
    test_seqs = read_test_qa(args.paragraphs, args.lines)
    m = lstm_models.load_model(args.model)
    evaluate_qa(m, test_seqs)


print("done.")
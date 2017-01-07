import itertools
from utils import *
import csv

prefixlen = 40 # prefixlen must be >= maxlen!

def get_training_seqs(f: typing.io.BinaryIO, lines: int) -> Iterator[Tuple[List[Word], List[Word]]]:
    'Returns list of sequences of words for training'
    if lines<0:
        return read_train_query_paras(f)
    else:
        return itertools.islice(read_train_query_paras(f), 0, lines)


# def get_test_seqs_next_word(f: typing.io.BinaryIO, lines: int) -> List[TestSeq]:
#     'Returns a list of ( sequences of words, next word )'
#     paras = [para for para in read_test_query_paras(lines)]
#     # Todo change from next word to next seq of words
#
#     result = []
#     for seq, truth, negatives in paras[0:lines]:
#         result.append(TestSeq(sequence=seq,
#                               truth=truth[0],
#                               candidates=set([truth[0]]+[negtext[0]
#                                               for negtext in negatives])))
#     return result


# def read_query_paras_with_negatives(f, lines: int = None) -> Iterator[Tuple[List[Word], List[Word], List[List[Word]]]]:
#     """ Read text of TREC-CAR paragraphs """
#
#     rows = csv.reader(f, delimiter='\t')
#     if lines is not None:
#         rows = itertools.islice(rows, 0, lines)
#     for row in rows:
#         page, sectionpath, text = row[0:3]
#         negtexts = row[4:]
#         sectionpath = filter_field(sectionpath)
#         text = filter_field(text)
#         negtexts = list(map(filter_field, negtexts))
#         if len(sectionpath) == 0 or len(text) == 0 or len(negtexts) == 0: continue
#         yield (sectionpath, text, negtexts)

def tokenize(text):
    text = nltk.tokenize.word_tokenize(text.lower())
    return list(filter(is_good_token, text))


def read_test_qa(f, lines:int) -> Iterator[TestSeq]:
    if lines<0:
        return read_test_qa_(f)
    else:
        return itertools.islice(read_test_qa_(f), 0, lines)

def read_test_qa_(f) -> Iterator[TestSeq]:
    """ Read text of TREC-CAR paragraphs from wikistein test format"""

    old_query_id = ""
    old_query_text = list()
    candidates = []
    for row in csv.reader(f, delimiter='\t'):
        query_id, page, sectionpath, paragraph_id, text, judgment = row
        if len(query_id) == 0 or len(text) == 0: continue

        if old_query_id == "":
            old_query_id = query_id
            old_query_text = tokenize(" ".join([page, sectionpath]))
        if query_id == old_query_id:
            candidates.append(TestCandidate(paragraph_id, tokenize(text)))
        else :
            print( old_query_id)
            yield (old_query_id, old_query_text, candidates)
            old_query_id = query_id
            old_query_text = tokenize(" ".join([page, sectionpath]))
            candidates.append(TestCandidate(paragraph_id, tokenize(text)))
    if len(candidates)>0:
        yield (old_query_id, old_query_text, candidates)

def read_train_query_paras(f) -> Iterator[Tuple[List[Word], List[Word]]]:
    """ Read text of TREC-CAR paragraphs from wikistein cluster format"""
    for row in csv.reader(f, delimiter='\t'):
        query_id, page, sectionpath, paragraph_id, text = row
        if len(query_id) == 0 or len(text) == 0: continue
        query_text = tokenize(" ".join([page, sectionpath]))
        yield (query_text, text)

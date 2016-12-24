import itertools
from utils import *
import csv

prefixlen = 40 # prefixlen must be >= maxlen!

def get_training_seqs(f: typing.io.BinaryIO, lines: int) -> Iterator[Tuple[List[Word],List[Word]]]:
    'Returns list of sequences of words for training'
    if lines<0:
        return read_query_paras(f)
    else:
        return list(itertools.islice(read_query_paras(f), 0, lines))


def get_test_seqs(f: typing.io.BinaryIO, lines:int) -> List[TestSeq]:
    'Returns a list of ( sequences of words, next word )'
    paras = [para for para in read_query_paras_with_negatives(f, lines)]
    # Todo change from next word to next seq of words

    result = []
    for seq, truth, negatives in paras[0:lines]:
        result.append(TestSeq(sequence=seq,
                              truth=truth[0],
                              candidates=[truth[0]]+[negtext[0] for negtext in negatives]))
    # result = [TestSeq(sequence=para[0], truth=para[1][0], candidates=[negtext[0] for negtext in para[2]]) for para in
    #           paras[0:lines]]
    return result


def read_query_paras_with_negatives(f, lines:int) -> Iterator[Tuple[List[Word], List[Word], List[List[Word]]]]:
    """ Read text of TREC-CAR paragraphs """

    for row in itertools.islice(csv.reader(f,delimiter='\t'), 0, lines):
        page, sectionpath, text = row[0:3]
        negtexts = row[4:]
        sectionpath = filter_field(sectionpath)
        text = filter_field(text)
        negtexts = map(filter_field, negtexts)
        if sectionpath == '' or text == '': continue
        yield (sectionpath, text, negtexts)

def filter_field(text):
    text = nltk.tokenize.word_tokenize(text.lower())
    return list(filter(is_good_token, text))


def read_query_paras(f) -> Iterator[Tuple[List[Word],List[Word]]]:
    """ Read text of TREC-CAR paragraphs """
    for row in csv.reader(f, delimiter='\t'):
        page, sectionpath, text = row
        query = filter_field(sectionpath.lower())
        text = filter_field(text.lower())
        if query == '' or text == '': continue
        yield (query, text)

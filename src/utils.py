import collections

from typing import Dict, Tuple, List, Set, TypeVar, Iterator, Generator
import typing

import nltk.tokenize
import numpy as np
from trec_car import read_data


QueryId = str
ParagraphId = str

Word = str
Char = str



TestSeqNextWord = typing.NamedTuple('TestSeqNextWord', [('sequence', List[Word]),
                                        ('truth', Word),
                                        ('candidates', Set[Word])])

TestCandidate = typing.NamedTuple('TestCandidate', [('paragraph_id', ParagraphId),
                                                    ('sequence', List[Word])])

TestSeq = typing.NamedTuple('TestSeq',[('query_id', QueryId),
                                          ('query', List[Word]),
                                          ('candidates',List[TestCandidate])])

Ranking = List[Tuple[Word, float]]
RankingWithTruth = Tuple[Ranking, Word]

stopwords = set(nltk.corpus.stopwords.words('english'))

def is_good_token(token: Word) -> bool:
    x = ''.join(list(filter(lambda c: str.isalpha(c), token)))
    if len(x) <= 2: return False
    if x in stopwords: return False
    return True

def download_nlk_resources():
    nltk.download()
import collections

import nltk.tokenize
import numpy as np
from trec_car import read_data

def read_glove(fname):
    """ Read GloVe embeddings """
    embeddings = {}
    for line in open(fname):
        xs = line.split()
        word = xs[0]
        vec = np.array(list(map(float, xs[1:])))
        embeddings[word] = vec
    return embeddings

class ExactEmbeddingIndex(object):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def get(self, vec):
        sim = None
        word = None
        for k,v in self.embeddings.items():
            tmp = np.dot(v, vec) / np.linalg.norm(vec) / np.linalg.norm(v)
            if sim is None or tmp > sim:
                word = k
                sim = tmp
        return word

class BinnedEmbeddingIndex(object):
    """
    An efficient structure for looking up the word associated with a particular
    embedding vector
    """
    def __init__(self, embeddings):
        print('preparing word index',)
        idx = collections.defaultdict(lambda: [])
        for k,v in embeddings.items():
            idx[tuple(x > 0 for x in v[:10])].append((k,v))
        self.word_index = idx
        print('done')

    def get(self, vec):
        word = None
        sim = None
        key = tuple(x > 0 for x in vec[:10])
        for k,v in self.word_index.get(key, {}):
            tmp = np.dot(v, vec) / np.linalg.norm(vec) / np.linalg.norm(v)
            if sim is None or tmp > sim:
                word = k
                sim = tmp
        return word

def read_paras():
    """ Read text of TREC-CAR paragraphs """
    paras = []
    for para in read_data.iter_paragraphs(open('data/release.paragraphs', 'rb')):
        text = ''
        for body in para.bodies:
            if isinstance(body, read_data.ParaText):
                text += body.text
            elif isinstance(body, read_data.ParaLink):
                text += body.anchor_text
        paras.append(nltk.tokenize.word_tokenize(text.lower()))
    return paras


def download_nlk_resources():
    nltk.download()
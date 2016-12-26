from utils import *

WordVector = np.ndarray

class Embeddings(object):
    def __init__(self, fname):
        self._embeddings = Embeddings._read_glove(fname)

    def get(self, word: Word):
        if word in self._embeddings:
            return self._embeddings.get(word)
        else :
            return np.ones(self.dim(), dtype=float)
            # todo fix this

    def lookup(self, vec: WordVector):
        raise NotImplementedError

    def dim(self):
        for x in self._embeddings.values():
            return len(x) # just the first element

    @staticmethod
    def _read_glove(fname):
        """ Read GloVe embeddings """
        embeddings = {}
        for line in open(fname):
            xs = line.split()
            word = xs[0]
            vec = np.array(list(map(float, xs[1:])))
            embeddings[word] = vec
        return embeddings

    def __contains__(self, word: Word):
        return True
        # Todo fix this!
        # return word in self._embeddings


class ExactEmbeddings(Embeddings):
    def __init__(self, fname):
        Embeddings.__init__(self, fname)

    def lookup(self, vec):
        sim = None
        word = None
        for k,v in self._embeddings.items():
            tmp = np.dot(v, vec) / np.linalg.norm(vec) / np.linalg.norm(v)
            if sim is None or tmp > sim:
                word = k
                sim = tmp
        return word


class BinnedEmbeddings(Embeddings):
    """
    An efficient structure for looking up the word associated with a particular
    embedding vector
    """
    def __init__(self, fname: str):
        Embeddings.__init__(self, fname)

        print('preparing word index',)
        idx = collections.defaultdict(lambda: [])
        for k,v in self._embeddings.items():
            idx[tuple(x > 0 for x in v[:10])].append((k,v))
        self.word_index = dict(idx) # convert to a dict to allow pickling
        print('done')

    def lookup(self, vec:np.ndarray):
        assert isinstance(vec,np.ndarray)
        assert len(vec)==self.dim(), len(vec)
        word = None
        sim = None
        key = tuple(x > 0 for x in vec[:10])
        for k,v in self.word_index.get(key, {}):
            tmp = np.dot(v, vec) / np.linalg.norm(vec) / np.linalg.norm(v)
            if sim is None or tmp > sim:
                word = k
                sim = tmp
        return word

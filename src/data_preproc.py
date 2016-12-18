import itertools
from utils import *

prefixlen = 40 # prefixlen must be >= maxlen!

def get_training_seqs(f, lines:int) -> List[List[Word]]:
    'Returns list of sequences of words for training'
    if lines<0:
        return read_paras(f)
    else:
        return list(itertools.islice(read_paras(f), 0, lines))



def get_test_seqs(f, lines:int) -> List[TestSeq]:
    'Returns a list of ( sequences of words, next word )'
    paras = [para for para in read_paras(f) if len(para)>prefixlen + 1]

    num_lines = len(paras) if lines<0 else min(lines, len(paras))

    result = []
    for i in range(1,num_lines -1):
        para = paras[i]
        cands = {paras[j][prefixlen] for j in {i-1, i, i+1}}

        result.append(TestSeq(sequence=para[0:prefixlen], truth=para[prefixlen], candidates=cands))
    return result

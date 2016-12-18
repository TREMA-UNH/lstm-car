from utils import *

prefixlen = 40 # prefixlen must be >= maxlen!

def get_training_seqs() -> List[List[Word]]:
    'Returns list of sequences of words for training'
    paras = read_paras()
    return paras


def get_test_seqs() -> List[TestSeq]:
    'Returns a list of ( sequences of words, next word )'
    paras = [para for para in read_paras() if len(para)>prefixlen + 1]

    result = []
    for i in range(1,len(paras)-1):
        para = paras[i]
        cands = {paras[j][prefixlen] for j in {i-1, i, i+1}}

        result.append(TestSeq(sequence=para[0:prefixlen], truth=para[prefixlen], candidates=cands))
    return result

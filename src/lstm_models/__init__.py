from utils import *

class ParaCompletionModel:
    def train(self, training_seqs: List[List[Word]]):
        raise NotImplementedError

    def generate_word(self, test_inputs: List[List[Word]]) -> List[Word]:
        raise NotImplementedError

    def rank_word(self,
                  test_seqs: List[TestSeq]) \
            -> List[RankingWithTruth]:
        raise NotImplementedError

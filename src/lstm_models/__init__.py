from utils import *

class ParaCompletionModel:
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    def load(things, path: str):
        raise NotImplementedError

    def save(self, path: str, things=None):
        import pickle, os.path
        if not os.path.isdir(path):
            os.makedirs(path)
        if things is None:
            things = {}
        things['type'] = self.__class__
        pickle.dump(things, open(os.path.join(path, 'type.pickle'), 'wb'))

    def train(self, training_seqs: List[List[Word]]):
        raise NotImplementedError

    def train_qa(self, training_pairs: List[Tuple[List[Word],List[Word]]]):
        raise NotImplementedError

    def generate_word(self, test_inputs: List[List[Word]]) -> List[Word]:
        raise NotImplementedError

    def rank_word(self,
                  test_seqs: List[TestSeq]) \
            -> List[RankingWithTruth]:
        raise NotImplementedError


def load_model(path: str) -> ParaCompletionModel:
    import pickle, os.path
    things = pickle.load(open(os.path.join(path, 'type.pickle'), 'rb'))
    ty = things['type']
    return ty.load(things, path)
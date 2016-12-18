from utils import *
from typing import Iterable

def mrr(rankings: List[RankingWithTruth]) -> float:
    results = []
    for (ranking, truth) in rankings:
        rr = 0.0
        for (rank, ( elem, score) ) in enumerate(ranking):
            if elem == truth:
                rr = 1/(rank+1)
        results.append(rr)

    return sum(results)/ len(results)

def prec1Pred(predicted: Iterable[Word], truths: Iterable[Word]) -> float:
    results = []
    for pred,truth in zip(predicted, truths):
        if pred == truth:
            results.append(1.0)
        else:
            results.append(0.0)

    return sum(results) / len(results)

def prec1Rank(rankings: List[RankingWithTruth]) -> float:
    predicted, truths = zip(*[(ranking[0][0], truth)
                              for (ranking, truth) in rankings])
    return prec1Pred(predicted, truths)


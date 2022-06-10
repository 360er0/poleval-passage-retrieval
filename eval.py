from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--true', dest='true', help='tsv file with true (question, passage) pairs', metavar='FILE')
    parser.add_argument('--pred', dest='pred', help='tsv file with predicte (question, passage) pairs', metavar='FILE')
    parser.add_argument('--k', type=int, dest='k', default=10)
    args = parser.parse_args()

    true = pd.read_csv(args.true, sep='\t')
    pred = pd.read_csv(args.pred, sep='\t')
    pred['score'] -= pred['score'].min() - 1e-6  # convert to non-negative scores

    scores = true.merge(pred, on=['question-id', 'passage-id'], how='outer', suffixes=('_true', '_pred'))
    scores = scores.fillna(0)
    scores = scores.groupby('question-id').agg(list)
    scores = scores.reset_index()

    ndcg = scores.apply(lambda r: ndcg_score([r['score_true']], [r['score_pred']], k=args.k), axis=1).mean()

    print(f'NDCG@{args.k}: {ndcg:.3f}')

import argparse
import os
import pickle as pkl
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from tqdm import tqdm

import sklearn

emb_path = '/Users/xujinghua/speaker-verification-with-NeMo/data/embeddings/embeddings_manifest_embeddings.pkl'

emb = pkl.load(open(emb_path, 'rb'))

# print(emb.keys())

X = emb['Users@xujinghua@jxu_en_1.wav']
Y = emb['Users@xujinghua@jxu_cn_1.wav']

# print(emb1)
# print(sklearn.metrics.pairwise.cosine_similarity(emb1, emb2))

score = (X @ Y.T) / (((X @ X.T) * (Y @ Y.T)) ** 0.5)
score = (score + 1) / 2

print(score)

# score: 0.9686643297704525

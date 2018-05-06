import numpy as np
import os
from collections import defaultdict
from torch.nn.modules import Embedding
import torch as th
from torch import nn


def from_pretrained(embeddings, freeze=True):
    assert embeddings.dim() == 2, \
        'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = nn.Parameter(embeddings)
    embedding.weight.requires_grad = not freeze
    return embedding


#TODO Do not use static methods
class WordVectorLoader:
    word2index = None
    index2word = None
    word_vec = None
    embeddings = None
    sense_num = None
    word_sim_adj = None

    def __init__(self):
        raise NotImplemented("Static class")

    @classmethod
    def build(cls, dwords, word_vec):
        cls.sense_num = min(dwords.values())
        index2word = [None] * (max(dwords.values()) + 1)
        for word, i in dwords.items():
            index2word[i] = word

        assert all((index2word[i] is None for i in range(cls.sense_num)))
        assert all((index2word[i] is not None for i in range(cls.sense_num, len(index2word))))
        cls.embeddings = from_pretrained(th.Tensor(word_vec), True)
        cls.index2word = index2word
        cls.word2index = dwords
        cls.word_vec = np.array(word_vec)
        cls.load_sim_adj()
        return cls.embeddings

    @classmethod
    def embed(cls, inputs):
        if cls.embeddings is None:
            raise ValueError()
        return cls.embeddings(inputs)

    @classmethod
    def load_sim_adj(cls):
        word_sim_adj = defaultdict(lambda: {})
        if not os.path.exists('all_nn.tsv'):
            print('cannot find all_nn')
            cls.word_sim_adj = word_sim_adj
            return
        with open('all_nn.tsv') as f:
            for line in f:
                w1, w2, sim = line[:-1].split('\t')
                if w1 not in cls.word2index or w2 not in cls.word2index:
                    continue
                sim = float(sim)
                i1, i2 = cls.word2index[w1], cls.word2index[w2]
                word_sim_adj[i1][i2] = sim
                word_sim_adj[i2][i1] = sim
        cls.word_sim_adj = dict(word_sim_adj)

    @classmethod
    def get_vec_by_word(cls, w):
        return cls.word_vec[WordVectorLoader.word2index[w] - cls.sense_num]

    @classmethod
    def get_vec_by_index(cls, i):
        return cls.word_vec[i]


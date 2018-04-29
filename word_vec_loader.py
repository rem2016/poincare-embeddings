import spacy
import numpy as np
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

    def __init__(self):
        raise NotImplemented("Static class")

    @classmethod
    def build(cls, dwords):
        print('Build embedding')
        cls.sense_num = min(dwords.values())
        nlp = spacy.load('en_core_web_lg')
        word_vec = [None] * len(dwords)
        index2word = [None] * len(dwords)
        for word, i in dwords.items():
            i -= cls.sense_num
            word_vec[i] = nlp(word).vector.tolist()
            index2word[i] = word

        assert all((x is not None for x in word_vec))
        print('loading embedding finished')
        cls.embeddings = from_pretrained(th.Tensor(word_vec), True)
        print('Load embedding from pre-trained finished')
        cls.index2word = index2word
        cls.word2index = dwords
        cls.word_vec = np.array(word_vec)
        return cls.embeddings

    @classmethod
    def embed(cls, inputs):
        if cls.embeddings is None:
            raise ValueError()
        return cls.embeddings(inputs)

    @classmethod
    def get_vec_by_word(cls, w):
        return cls.word_vec[WordVectorLoader.word2index[w]]

    @classmethod
    def get_vec_by_index(cls, i):
        return cls.word_vec[i]


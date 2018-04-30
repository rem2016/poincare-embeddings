import torch as th
from nltk.corpus import brown, treebank
from nltk.corpus import wordnet as wn
import sys
from sematch.evaluation import WordSimEvaluation
from nltk.stem import WordNetLemmatizer
import numpy as np
from numpy.linalg import norm
from nltk.corpus import wordnet as wn
from sematch.utility import memoized


class Evaluator:
    data_word_noun = ['noun_rg', 'noun_mc', 'noun_ws353', 'noun_ws353-sim', 'noun_simlex']
    sim_methods_noun = ['path', 'lch', 'wup', 'li', 'res', 'lin', 'jcn', 'wpath']
    ws_eval = WordSimEvaluation()
    _wn_lemma = WordNetLemmatizer()

    def __init__(self, embs, objs):
        self.synmap = self.load(embs, objs)

    def load(self, embs, objs):
        return {wn.synset(objs[i]): emb for i, emb in enumerate(embs)}

    def evaluate(self, method='tanh'):
        sim = lambda x, y: self.word_similarity(x, y, method)
        cors = [self.ws_eval.evaluate_metric('poincare', sim, dset_name, save_results=True) for dset_name in
                self.data_word_noun]
        cors = {name: cor for name, cor in zip(self.data_word_noun, cors)}
        return cors

    @memoized
    def word_similarity(self, w1, w2, method='exp'):
        s1 = self.word2synset(w1)
        s2 = self.word2synset(w2)
        return self.max_synset_similarity(s1, s2, self.syn_similarity_gen(method))

    def syn_similarity(self, s1, s2):
        dist = self.get_dist(self.synmap[s1], self.synmap[s2])
        return -dist

    def syn_similarity_gen(self, dist2sim):
        mapping_methods = {
            'reciprocal': lambda x: 1 / (1 + x),
            'neg': lambda x: (33 - x) / 33,
            'exp': lambda x: np.exp(-x),
            'tanh': lambda x: 2 - 2 / (1 + np.exp(-x)),
        }

        if dist2sim not in mapping_methods:
            raise NotImplementedError()

        def sim(x, y):
            try:
                dist = self.get_dist(self.synmap[x], self.synmap[y])
            except KeyError:
                return 0
            _sim = mapping_methods[dist2sim](dist)
            return _sim

        return sim

    def word2synset(self, word, pos=wn.NOUN):
        word = self._wn_lemma.lemmatize(word)
        return wn.synsets(word, pos)

    def get_dist(self, v1, v2):
        s = norm((v1 - v2)) ** 2
        m = (1 - norm(v1) ** 2) * (1 - norm(v2) ** 2)
        ans = np.arccosh(1 + 2 * s / m)
        return ans

    def max_synset_similarity(self, syns1, syns2, sim_metric):
        """
        Compute the maximum similarity score between two list of synsets
        :param syns1: synset list
        :param syns2: synset list
        :param sim_metric: similarity function
        :return: maximum semantic similarity score
        """
        return max([sim_metric(c1, c2) for c1 in syns1 for c2 in syns2] + [0])

    @staticmethod
    def initialize_by_file(fin):
        with open(fin, 'rb') as f:
            model = th.load(f)
        return Evaluator(model['model']['lt.weight'], model['objects'])

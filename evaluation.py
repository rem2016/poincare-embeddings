import torch as th
from nltk.corpus import brown, treebank
from nltk.corpus import wordnet as wn
import sys
import data
from sematch.evaluation import WordSimEvaluation
from model import SNEmbedding, PoincareDistance
from nltk.stem import WordNetLemmatizer
import numpy as np
from numpy.linalg import norm
from nltk.corpus import wordnet as wn
from sematch.utility import memoized


mapping_methods = {
    'reciprocal': lambda x: 1 / (1 + x),
    'neg': lambda x: (33 - x) / 33,
    'exp': lambda x: np.exp(-x),
    'tanh': lambda x: 2 - 2 / (1 + np.exp(-x)),
    'cos': lambda x, y: np.sum(x*y, axis=-1) / (norm(x) * norm(y))
}


class NoExistError(ValueError):
    pass


class OutofdateVersionError(ValueError):
    pass


class Evaluator:
    data_word_noun = ['noun_rg', 'noun_mc', 'noun_ws353', 'noun_ws353-sim', 'noun_simlex']
    sim_methods_noun = ['path', 'lch', 'wup', 'li', 'res', 'lin', 'jcn', 'wpath']
    ws_eval = WordSimEvaluation()
    _wn_lemma = WordNetLemmatizer()

    def __init__(self, embs, objs, dict=None, index2word=None, k=None):
        """

        :param embs: Embedding
        :param objs: index -> sense
        :param dict: model dict_state
        :param index2word: index -> word (start from 0)
        """
        self.failed_times = 0
        if k is None:
            k = 12
        self.k = k
        self.synmap = None
        self.model = None
        self.dict_state = dict
        self.index2word = index2word
        self.embs = embs
        self.objects = objs
        self.word2vec = None
        self.test_set = None

    def load(self, embs, objs):
        return {wn.synset(objs[i]): emb for i, emb in enumerate(embs) if i < len(objs)}

    def load_wordvec(self):
        if self.word2vec is not None:
            return

        word2vec = {}
        for i, name in enumerate(self.index2word[len(self.objects):], len(self.objects)):
            word2vec[name] = self.embs[i]

        self.word2vec = word2vec

    def evaluate(self, method='tanh', is_word_level=False, try_use_word=False):
        self.failed_times = 0
        if try_use_word:
            if self.index2word is not None:
                is_word_level = True
                print('Word Level')
            else:
                print('Sense Level')
        if self.synmap is None:
            self.synmap = self.load(self.embs, self.objects)
        if is_word_level:
            if self.word2vec is None:
                try:
                    self.load_wordvec()
                except Exception:
                    raise OutofdateVersionError()

            def sim(x, y):
                v = self.word_similarity(x, y, method, True)
                if isinstance(v, th.Tensor):
                    v = float(v.item())
                return v
        else:
            def sim(x, y):
                v = self.word_similarity(x, y, method)
                if isinstance(v, th.Tensor):
                    v = float(v.item())
                return v
        cors = [self.ws_eval.evaluate_metric('poincare', sim, dset_name, save_results=True) for dset_name in
                self.data_word_noun]
        cors = {name: cor for name, cor in zip(self.data_word_noun, cors)}
        return cors

    @memoized
    def word_similarity(self, w1, w2, method='exp', is_word=False):
        if not is_word:
            s1 = self.word2synset(w1)
            s2 = self.word2synset(w2)
            return self.max_synset_similarity(s1, s2, self.syn_similarity_gen(method))
        try:
            if method == 'cos':
                return mapping_methods['cos'](self.word2vec[w1], self.word2vec[w2])
            dist = self.word_dist(w1, w2) * self.k
            return mapping_methods[method](dist)
        except (NoExistError, KeyError):
            self.failed_times += 1
            return 0
            s1 = self.word2synset(w1)
            s2 = self.word2synset(w2)
            return self.max_synset_similarity(s1, s2, self.syn_similarity_gen(method))

    def word_dist(self, w1, w2):
        if w1 not in self.word2vec or w2 not in self.word2vec:
            raise NoExistError()
        return self.get_dist(self.word2vec[w1], self.word2vec[w2])

    def syn_similarity(self, s1, s2):
        dist = self.get_dist(self.synmap[s1], self.synmap[s2])
        return -dist

    def syn_similarity_gen(self, dist2sim):
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
        if isinstance(v1, np.ndarray):
            v1 = th.from_numpy(v1)
            v2 = th.from_numpy(v2)
        return PoincareDistance()(v1, v2)

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
    def initialize_by_file(fin, k=None):
        with open(fin, 'rb') as f:
            model = th.load(f)
        return Evaluator(model['model']['lt.weight'], model['objects'], model['model'], model.get('word_index', None), k=k)

    def rank(self):
        # rank rank rank rank
        # how to rank???
        # OK, first ! loading the model.
        # But it's already done ....
        # then.. second! we can.. we can load the test set!
        # yes... that is the point
        # here we go!
        self.load_test_set()
    # excuse me ? so ???
    # yeah, you implement that function
    # ..... fine you win
    # The song <The light that never come> is AWwwwwwwwwwwwwwwwwwwwwESOME
    # Enjoy yourself.
    # You can simply use the slurp func, dud

    # I fixed a fatal bug!!!   Oh!!!!!!!!!!!!!
        from embed import ranking

        # ranking(self.test_set, self.)
        # I ... think we should construct the model first
        self.load_model()
        # So easy to use. I cannot believe it
        return ranking(self.test_set, self.model, PoincareDistance)  # Mean rank, Mean ap

    def load_test_set(self):
        if self.test_set is not None:
            return
        from embed import get_adjacency_by_idx
        idx, _, _ = data.slurp('wordnet/noun_closure.train.tsv', objects=self.objects)
        adj = get_adjacency_by_idx(idx)
        self.test_set = adj

    def load_model(self):
        if self.model is not None:
            return

        model = SNEmbedding(
            len(self.objects),
            self.embs.size(1),
            dist=PoincareDistance,
            max_norm=1
        )

        model.load_state_dict(self.dict_state)
        self.model = model


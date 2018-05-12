import numpy as np
from numpy.random import choice, randint
import random
import copy
import torch as th
from model import SNEmbedding
from torch import nn
from numpy.linalg import norm
from word_vec_loader import WordVectorLoader
from torch.autograd import Function, Variable
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import copy

eps = 1e-5


class GraphDataset(Dataset):
    _ntries = 10
    _dampening = 1

    def __init__(self, idx, objects, nnegs, unigram_size=1e8):
        print('Indexing data')
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects
        self.max_tries = self.nnegs * self._ntries

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(len(objects), dtype=np.float)
        nents = 0
        for i in range(idx.shape[0]):
            t, h, w = self.idx[i]
            self._counts[h] += w
            self._weights[t][h] += w
            nents = max((nents, t, h))
        self._weights = dict(self._weights)
        nents += 1
        if len(objects) != nents:
            assert WordVectorLoader.word2index is not None, \
                f'Number of objects do no match {len(objects)} != {nents}'
            exp_num = len(objects) + len(WordVectorLoader.word2index)
            assert exp_num == nents, \
                f'Number of objects do no match {exp_num} != {nents}'
        self.total_nodes_num = nents

        if unigram_size > 0:
            c = self._counts ** self._dampening
            self.unigram_table = choice(
                len(objects),
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return len(self.idx)

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return Variable(th.cat(inputs, 0)), Variable(th.cat(targets, 0))


class SNGraphDataset(GraphDataset):
    model_name = '%s_%s_dim%d'

    @staticmethod
    def make(t, h, size, adj, nnegs, max_tries=500):
        negs = set()
        ntries = 0
        while ntries < max_tries and len(negs) < nnegs:
            n = randint(0, size)
            if n not in adj[t]:
                negs.add(n)
            ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()

    def __getitem__(self, i):
        t, h, _ = [int(x) for x in self.idx[i]]
        nnegs = self.nnegs
        if self.burnin:
            nnegs *= 0.2
        return self.make(t,
                         h,
                         self.total_nodes_num,
                         self._weights,
                         self.nnegs,
                         self.max_tries)

    @classmethod
    def initialize(cls, distfn, opt, idx, objects, max_norm=1, node_num=None):
        if node_num is None:
            node_num = len(objects)
        conf = []
        model_name = cls.model_name % (opt.dset, opt.distfn, opt.dim)
        data = cls(idx, objects, opt.negs)
        model = SNEmbedding(
            node_num,
            opt.dim,
            dist=distfn,
            max_norm=max_norm
        )
        data.objects = objects
        return model, data, model_name, conf

    @classmethod
    def initialize_word2vec_nn(cls, distfn, opt, idx, objects):
        import join_word2vec
        conf = []
        model_name = cls.model_name % (opt.dset, opt.distfn, opt.dim)
        data = cls(idx, objects, opt.negs)
        model = join_word2vec.SNEmbeddingWithWord(
            len(data.objects),
            opt.dim,
            sense_num=len(objects),
            hidden_size=opt.nn_hidden_size,
            layer_size=opt.nn_hidden_layer
        )
        data.objects = objects
        return model, data, model_name, conf


class OnlySynsetDataset(SNGraphDataset):
    def __init__(self, idx, objects, nnegs, unigram_size=1e8):
        _idx = []
        for h, t, w in idx:
            if h >= len(objects) or t >= len(objects):
                continue
            _idx.append([h, t, w])
        _idx = np.array(_idx)
        super().__init__(idx, objects, nnegs, unigram_size)

    @classmethod
    def initialize(cls, distfn, opt, idx, objects, max_norm=1, node_num=None):
        if node_num is None:
            node_num = len(objects)
        conf = []
        model_name = cls.model_name % (opt.dset, opt.distfn, opt.dim)
        data = cls(idx, objects, opt.negs)
        model = SNEmbedding(
            node_num,
            opt.dim,
            dist=distfn,
            max_norm=max_norm
        )
        data.objects = objects
        return model, data, model_name, conf

    @classmethod
    def initialize_word2vec_nn(cls, distfn, opt, idx, objects):
        import join_word2vec
        conf = []
        model_name = cls.model_name % (opt.dset, opt.distfn, opt.dim)
        data = cls(idx, objects, opt.negs)
        model = join_word2vec.SNEmbeddingWithWord(
            len(data.objects),
            opt.dim,
            sense_num=len(objects),
            hidden_size=opt.nn_hidden_size,
            layer_size=opt.nn_hidden_layer
        )
        data.objects = objects
        return model, data, model_name, conf


def is_good_sim(sim):
    # return sim < -0.2 or sim > 0.3  # total 5%
    return sim < -0.1 or sim > 0.2  # total 20%


def is_perfect_sim(sim):
    return sim < -0.25 or sim > 0.4  # total 1%


class WordsDataset(Dataset):
    # pay attention to sense num
    # except from output, all index is based on word vec !!
    def __init__(self, word_vec: np.array,
                 sense_num: int,
                 sim_adj: dict,
                 link_adj: dict,
                 pair_per_word: int = 100,
                 max_pairs=200):
        self.npair = max(pair_per_word, max_pairs)
        self.word_vec = np.array(word_vec)
        self.sense_num = sense_num
        self.least_pos = max(self.npair // 5, 1)
        self.word_num = len(word_vec)
        self.valid_index = []
        self.max_pairs = max_pairs
        self.adj = [{} for _ in range(len(word_vec))]
        self.init_adj(sim_adj)
        self.link_adj = link_adj

    def __calc_dist(self, a, b):
        sim = np.sum(a * b, axis=-1) / (norm(a, axis=-1) * norm(b, axis=-1))
        return sim

    def __get_word(self, index):
        b_indexes = list(np.random.choice(self.word_num, self.npair))
        a_v = np.expand_dims(self.word_vec[index], 0)
        b_vs = self.word_vec[b_indexes]
        sim = list(self.__calc_dist(a_v, b_vs))
        for i, v in enumerate(sim):
            if is_perfect_sim(v):
                self.adj[index][b_indexes[i]] = v
                self.adj[b_indexes[i]][index] = v

        if len(self.adj[index]):
            adj_num = self.max_pairs - self.npair
            if len(self.adj[index]) > adj_num > 0:
                used = set(np.random.choice(len(self.adj[index]), adj_num))
                for i, (b_index, _sim) in enumerate(self.adj[index].items()):
                    if i in used and b_index not in b_indexes:
                        b_indexes.append(b_index)
                        sim.append(_sim)
            else:
                b_indexes.extend(self.adj[index].keys())
                sim.extend(self.adj[index].values())
        return th.LongTensor([[index + self.sense_num, b_index + self.sense_num] for b_index in b_indexes]), \
               th.Tensor(sim)

    def __get_link(self, index):
        index = index + self.sense_num
        linked = self.link_adj[index]
        nodes_data = [SNGraphDataset.make(index,
                                          link,
                                          self.sense_num + self.word_num,
                                          self.link_adj,
                                          nnegs=20)
                      for link in linked]
        inputs, targets = zip(*nodes_data)
        return Variable(th.cat(inputs, 0)), Variable(th.cat(targets, 0))

    def __getitem__(self, index):
        words_input, words_target = self.__get_word(index)
        links_input, links_target = self.__get_link(index)
        return words_input, words_target, links_input, links_target

    def init_adj(self, sim_adj):
        for i in range(len(self.word_vec)):
            v = self.word_vec[i]
            if norm(v).item() < 1e-7:
                continue
            self.valid_index.append(i)

        for a, items in sim_adj.items():
            for b, sim in items.items():
                self.adj[a - self.sense_num][b - self.sense_num] = sim
                self.adj[b - self.sense_num][a - self.sense_num] = sim

    def calc_word_average_adj(self):
        num = 0
        for d in self.adj:
            num += len(d)
        return num / len(self.adj)

    def __len__(self):
        return self.word_num

    @classmethod
    def collate(cls, batch):
        inputs_w, targets_w, inputs_n, targets_n = zip(*batch)
        return Variable(th.cat(inputs_w, 0)), Variable(th.cat(targets_w, 0)), \
               Variable(th.cat(inputs_n, 0)), Variable(th.cat(targets_n, 0))


class WordAsHeadDataset(GraphDataset):
    model_name = '%s_%s_head_dim%d'

    def __init__(self, idx, objects, nnegs, unigram_size=1e8, sense_num=None):
        super().__init__(idx, objects, nnegs, unigram_size)
        if sense_num is None:
            raise ValueError("You should give me a sign about sense num bro")
        self.sense_num = sense_num
        self.idx = [(int(t), int(h), 1) for t, h, _ in self.idx
                    if int(t) >= self.sense_num > int(h)]

    def __getitem__(self, i):
        t, h, _ = [int(x) for x in self.idx[i]]
        negs = set()
        ntries = 0
        nnegs = self.nnegs
        if self.burnin:
            nnegs *= 0.1
        while ntries < self.max_tries and len(negs) < nnegs:
            n = randint(0, self.sense_num)
            if n not in self._weights[t]:
                negs.add(n)
            ntries += 1

        if len(negs) == 0:
            for i in range(self.sense_num):
                if i not in self._weights[t]:
                    negs.add(i)
                    break

        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()


class WordAsNegDataset(GraphDataset):
    model_name = '%s_%s_head_dim%d'

    def __init__(self, idx, objects, nnegs, unigram_size=1e8, words_num=None):
        super().__init__(idx, objects, nnegs, unigram_size)
        if words_num is None:
            raise ValueError("You should give me a sign about sense num bro")
        self.words_num = words_num
        self.sense_num = len(objects)
        self.idx = [(int(t), int(h), 1) for t, h, _ in self.idx
                    if int(t) < self.sense_num and int(h) < self.sense_num]

    def __getitem__(self, i):
        t, h, _ = self.idx[i]
        negs = set()
        ntries = 0
        nnegs = self.nnegs
        if self.burnin:
            nnegs *= 0.1
        while ntries < self.max_tries and len(negs) < nnegs:
            n = randint(self.sense_num, self.sense_num + self.words_num)
            if n not in self._weights[t]:
                negs.add(n)
            ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()

    def __len__(self):
        return len(self.idx) // self.nnegs

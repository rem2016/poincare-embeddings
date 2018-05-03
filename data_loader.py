import numpy as np
from numpy.random import choice, randint
import torch as th
from model import SNEmbedding
from torch import nn
from numpy.linalg import norm
from word_vec_loader import WordVectorLoader
from torch.autograd import Function, Variable
from torch.utils.data import Dataset
from collections import defaultdict as ddict

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

    def __getitem__(self, i):
        t, h, _ = [int(x) for x in self.idx[i]]
        negs = set()
        ntries = 0
        nnegs = self.nnegs
        if self.burnin:
            nnegs *= 0.1
        while ntries < self.max_tries and len(negs) < nnegs:
            if self.burnin:
                n = randint(0, len(self.unigram_table))
                n = int(self.unigram_table[n])
            else:
                n = randint(0, len(self.objects))
            if n not in self._weights[t]:
                negs.add(n)
            ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()

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


class WordsDataset(Dataset):
    def __init__(self, word_vec: np.array, sense_num, pair_per_word: int=10):
        self.npair = pair_per_word
        self.word_vec = word_vec
        self.sense_num = sense_num
        self.word_num = len(word_vec)

    def __getitem__(self, index):
        a_index = index // self.npair
        b_index = randint(0, self.word_num)
        a = self.word_vec[a_index]
        b = self.word_vec[b_index]
        sim = float(np.sum(a * b) / (norm(a) * norm(b)))
        return th.LongTensor([[a_index + self.sense_num, b_index + self.sense_num]]), \
               th.Tensor([sim]).view(1, )

    def __len__(self):
        if self.npair > self.word_num:
            return self.word_num ** 2
        return self.word_num * self.npair


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

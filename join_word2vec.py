import gc
from word_vec_loader import WordVectorLoader
import timeit
import numpy as np
from model import PoincareDistance
from torch.utils.data import DataLoader
import torch as th
from torch import nn


class WordModel(nn.Module):
    embedding_dim = 300

    def __init__(self, hidden_size, output_size, layer_size=1, word2vec=WordVectorLoader):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_layers = [nn.Linear(self.embedding_dim, hidden_size, bias=True)] +\
                             [nn.Linear(hidden_size, hidden_size, bias=True) for _ in range(layer_size - 2)] +\
                             [nn.Linear(hidden_size, output_size + 1, bias=True)]
        self.activates = [nn.Tanh() for _ in range(layer_size + 1)]
        self.word2vec = word2vec
        self.word_num = len(word2vec.word2index)
        self.init_weights()

    def init_weights(self, scale=1e-1):
        for layer in self.linear_layers:
            layer.state_dict()['weight'].uniform_(-scale, scale)

    def forward(self, word_index):
        is_single_input = len(word_index.size()) == 0
        if is_single_input:
            word_index = word_index.view(1, )
        assert len(word_index.size()) == 1
        v = self.word2vec.embeddings(word_index)
        for act, layer in zip(self.activates, self.linear_layers):
            v = act(layer(v))

        output = v.narrow(1, 0, self.output_size)
        radius = v.narrow(1, self.output_size, 1)
        norm = th.norm(output, 2, 1, keepdim=False)
        output = output / norm.view(-1, 1) * radius.view(-1, 1)
        if is_single_input:
            output = output.squeeze()

        return output

    def embedding(self):
        # cache the result may be even slower
        with th.no_grad():
            index = list(range(self.word_num))
            return self.forward(th.LongTensor(index)).numpy()


class EmbeddingWithWord(nn.Module):
    def __init__(self, size, dim, sense_num, dist=PoincareDistance,
                 max_norm=1, hidden_size=100, layer_size=1):
        super(EmbeddingWithWord, self).__init__()
        self.dim = dim
        self.lt = nn.Embedding(
            sense_num, dim,
            max_norm=max_norm,
            sparse=True,
            scale_grad_by_freq=False
        )
        self.sense_num = sense_num
        self.czx = WordModel(hidden_size=hidden_size, output_size=dim, layer_size=layer_size)
        self.dist = dist
        self.init_weights()

    def init_weights(self, scale=1e-4):
        self.lt.state_dict()['weight'].uniform_(-scale, scale)

    def forward(self, inputs):
        e = self.embed(inputs)
        fval = self._forward(e)
        return fval

    def sense_embedding(self):
        return list(self.lt.parameters())[0].data.cpu().numpy()

    def word_embedding(self):
        return self.czx.embedding()

    def embedding(self):
        sense = self.sense_embedding()
        word = self.word_embedding()
        return np.concatenate((sense, word), axis=0)

    def embed(self, inputs):
        size = list(inputs.size())
        inputs = inputs.view(-1)
        es = [self.lt(x) if x < self.sense_num else self.czx(x - self.sense_num) for x in inputs]
        e = th.stack(es)
        return e.view(size + [self.dim])


class SNEmbeddingWithWord(EmbeddingWithWord):
    def __init__(self, size, dim, sense_num, hidden_size=100, layer_size=1):
        super(SNEmbeddingWithWord, self).__init__(size, dim, sense_num,
                                                  PoincareDistance, 1,
                                                  hidden_size=hidden_size,
                                                  layer_size=layer_size)
        self.lossfn = nn.CrossEntropyLoss

    def _forward(self, e):
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.dist()(s, o).squeeze(-1)
        return -dists

    def loss(self, preds, targets, weight=None, size_average=True):
        lossfn = self.lossfn(size_average=size_average, weight=weight)
        return lossfn(preds, targets)


def calc_pair_sim(pairs):
    def _dist(v1, v2):
        return PoincareDistance()(v1, v2)

    def _dist2sim(d):
         return 2 - 2 / (1 + th.exp(-d))

    return th.cat([_dist2sim(_dist(pair[0], pair[1])) for pair in pairs])


def train(model, data, words_data, optimizer, opt, log, rank=1, queue=None):
    # setup parallel data loader
    loader = DataLoader(
        data,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )

    words_loader = DataLoader(
        words_data,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )

    for epoch in range(opt.epochs):
        epoch_loss = []
        loss = None
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * 0.01
            if rank == 1:
                log.info(f'Burnin: lr={lr}')

        for inputs, targets in loader:
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data[0])

        for inputs, targets in words_loader:
            elapsed = timeit.default_timer() - t_start
            optimizer.zero_grad()
            dists = calc_pair_sim(model.embed(inputs.squeeze()).view(len(targets), 2))
            loss = nn.MSELoss()(dists, targets)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data[0])

        if rank == 1:
            emb = None
            if epoch == (opt.epochs - 1) or epoch % opt.eval_each == (opt.eval_each - 1):
                emb = model
            if queue is not None:
                queue.put(
                    (epoch, elapsed, np.mean(epoch_loss), emb)
                )
            else:
                log.info(
                    'info: {'
                    f'"elapsed": {elapsed}, '
                    f'"loss": {np.mean(epoch_loss)}, '
                    '}'
                )
        gc.collect()

import gc
from word_vec_loader import WordVectorLoader
import timeit
import numpy as np
from model import PoincareDistance
from torch.utils.data import DataLoader
import torch as th
from torch import nn

if th.cuda.is_available():
    device = th.device('cuda')
else:
    device = th.device('cpu')


class WordModel(nn.Module):
    embedding_dim = 300

    def __init__(self, hidden_size, output_size, layer_size=1, word2vec=WordVectorLoader):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear_layers = [nn.Linear(self.embedding_dim, hidden_size, bias=True).to(device)] +\
                             [nn.Linear(hidden_size, hidden_size, bias=True).to(device) for _ in range(layer_size - 2)] +\
                             [nn.Linear(hidden_size, output_size + 1, bias=True).to(device)]
        self.word2vec = word2vec
        self.word_num = len(word2vec.word2index)
        self.init_weights()
        word_index = th.LongTensor(list(range(self.word_num)))
        self.cached = self._s_forward(word_index)

    def init_weights(self, scale=1e-1):
        for layer in self.linear_layers:
            layer.state_dict()['weight'].uniform_(-scale, scale)

    def _s_forward(self, word_index):
        size = list(word_index.size())
        word_index = word_index.contiguous().view(-1).to(device)
        v = self.word2vec.embeddings(word_index)
        for layer in self.linear_layers:
            v = layer(v).tanh()

        output = v.narrow(1, 0, self.output_size)
        radius = v.narrow(1, self.output_size, 1)
        norm = th.norm(output, 2, 1, keepdim=False)
        output = output / norm.view(-1, 1) * radius.view(-1, 1)
        output = output.view(*(size + [self.output_size]))
        return output

    def forward(self, word_index, no_grad=False):
        if no_grad:
            with th.no_grad():
                return self.cached[word_index.view(-1)].view(*(word_index.size() + [self.output_size]))
        else:
            output = self._s_forward(word_index)
            self.cached[word_index.contiguous().view(-1)] = output.view(-1, self.output_size)
        return output

    def embedding(self):
        with th.no_grad():
            index = th.LongTensor(list(range(self.word_num)))
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
        self.czx = WordModel(hidden_size=hidden_size, output_size=dim, layer_size=layer_size).to(device)
        self.dist = dist
        self.init_weights()

    def init_weights(self, scale=1e-4):
        self.lt.state_dict()['weight'].uniform_(-scale, scale)

    def forward(self, inputs, fix_nn=True, embed_index: tuple=None):
        e = self.embed(inputs, fix_nn, embed_index)
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

    def embed(self, inputs, no_grad: bool, embed_slice: tuple=None):
        if no_grad:
            e = self.embed_with_nn_fixed(inputs)
        else:
            e = self.embed_with_slice(inputs, embed_slice)
        return e

    def embed_with_slice(self, inputs, embed_slice):
        assert len(embed_slice) == 3
        dim, start, length = embed_slice
        e = inputs.narrow(*embed_slice)
        e = self.czx(e - self.sense_num)
        if start > 0:
            b = inputs.narrow(dim, 0, start)
            b = self.lt(b)
            e = th.cat((b, e), dim)
        if start + length < inputs.size()[dim]:
            b = inputs.narrow(dim, start + length, inputs.size()[dim] - start - length)
            b = self.lt(b)
            e = th.cat((e, b), dim)
        return e

    def embed_with_nn_fixed(self, inputs):
        size = list(inputs.size())
        inputs = inputs.view(-1)
        es = [self.lt(x) if x < self.sense_num else self.czx(x - self.sense_num, no_grad=True) for x in inputs]
        e = th.stack(es)
        e = e.view(size + [self.dim])
        return e


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


def calc_pair_sim(pairs, ll):
    def _dist(v1, v2):
        return PoincareDistance()(v1, v2)

    def _dist2sim(d):
        d = ll(d)
        return 2 - 2 / (1 + th.exp(-d))

    assert len(pairs.size()) == 3 and pairs.size(1) == 2
    return _dist2sim(_dist(pairs.narrow(1, 0, 1), pairs.narrow(1, 1, 1))).squeeze()


def combine_w2v_sim_train(model, data, words_data, optimizer, opt, log, rank=1, queue=None):
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
        batch_size=100,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )

    loss_balance = 1.0
    if opt.cold:
        loss_balance *= 0.1
    for epoch in range(opt.epochs):
        epoch_loss = []
        epoch_words_loss = []
        loss = None
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * 0.01
            if rank == 1:
                log.info(f'Burnin: lr={lr}')
        elif epoch == opt.burnin:
            loss_balance = 1.0

        for inputs, targets in loader:
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data.item())

        for inputs, targets in words_loader:
            elapsed = timeit.default_timer() - t_start
            model.zero_grad_kb()
            optimizer.zero_grad()
            dists = model.calc_pair_sim(inputs, opt.mapping_func)
            loss = nn.MSELoss()(dists, targets) * loss_balance
            loss.backward()
            optimizer.step(lr=lr)
            model.update_kb(lr=lr)
            epoch_words_loss.append(loss.data.item())

        if rank == 1:
            word_sim_loss = np.mean(epoch_words_loss) if len(epoch_words_loss) else None
            emb = None
            if epoch == (opt.epochs - 1) or epoch % opt.eval_each == (opt.eval_each - 1):
                emb = model
            if queue is not None:
                queue.put(
                    (epoch, elapsed, np.mean(epoch_loss), emb, word_sim_loss)
                )
            else:
                log.info(
                    'info: {'
                    f'"elapsed": {elapsed}, '
                    f'"loss": {np.mean(epoch_loss)}, '
                    f'"words_loss": {word_sim_loss}'
                    '}'
                )
            log.info('info: {'
                     f'"k": {model.k.item()}'
                     '}')

        if not opt.nobalance:
            if epoch >= opt.burnin * opt.balance_stage:
                loss_balance *= np.mean(epoch_loss) / np.mean(epoch_words_loss)
                if rank == 1:
                    log.info(f'Loss balance: {loss_balance}')
        gc.collect()

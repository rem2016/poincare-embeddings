#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch as th
from torch import nn
from join_word2vec import calc_pair_sim
import time
from word_vec_loader import WordVectorLoader
import timeit
from torch.utils.data import DataLoader
import gc


_lr_multiplier = 0.01


def train_mp(model, data, optimizer, opt, log, rank, queue):
    try:
        train(model, data, optimizer, opt, log, rank, queue)
    except Exception as err:
        log.exception(err)
        queue.put(None)


def train(model, data, optimizer, opt, log, rank=1, queue=None):
    # setup parallel data loader
    loader = DataLoader(
        data,
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
            lr = opt.lr * _lr_multiplier
            if rank == 1:
                log.info(f'Burnin: lr={lr}')
        for inputs, targets in loader:
            elapsed = timeit.default_timer() - t_start
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data.item())
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


class SingleThreadHandler:

    def __init__(self, log, train_types, test_types, data, fout, distfn, ranking):
        self.log = log
        self.types = train_types
        self.data = data
        self.fout = fout
        self.distfn = distfn
        self.ranking = ranking
        self.min_rank = (np.Inf, -1)
        self.max_map = (0, -1)

    def handle(self, msg):
        log = self.log
        types = self.types
        data = self.data
        fout = self.fout
        distfn = self.distfn
        ranking = self.ranking

        epoch, elapsed, loss, model = msg
        if model is not None:
            # save model to fout
            _fout = f'{fout}/{epoch}.nth'
            log.info(f'Saving model f{_fout}')
            th.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'objects': data.objects,
            }, _fout)

            # compute embedding quality
            log.info('Computing ranking')
            _start_time = time.time()
            mrank, mAP = ranking(types, model, distfn)
            log.info(f'Computing finished. Used time: {time.time() - _start_time}')
            if mrank < self.min_rank[0]:
                self.min_rank = (mrank, epoch)
            if mAP > self.max_map[0]:
                self.max_map = (mAP, epoch)
            log.info(
                ('eval: {'
                 '"epoch": %d, '
                 '"elapsed": %.2f, '
                 '"loss": %.3f, '
                 '"mean_rank": %.2f, '
                 '"mAP": %.4f, '
                 '"best_rank": %.2f, '
                 '"best_mAP": %.4f}') % (
                    epoch, elapsed, loss, mrank, mAP, self.min_rank[0], self.max_map[0])
            )
        else:
            log.info(f'json_log: {{"epoch": {epoch}, "loss": {loss}, "elapsed": {elapsed}}}')


def single_thread_train(model, data, optimizer, opt, log, handler, words_data=None):
    # setup parallel data loader
    loader = DataLoader(
        data,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )

    words_loader = None
    if words_data is not None:
        words_loader = DataLoader(
            words_data,
            batch_size=opt.batchsize,
            shuffle=True,
            num_workers=opt.ndproc,
            collate_fn=data.collate
        )

    for epoch in range(opt.epochs):
        epoch_loss = []
        epoch_words_loss = []
        loss = None
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier
            log.info(f'Burnin: lr={lr}')
        elapsed = 0
        for inputs, targets in loader:
            elapsed = timeit.default_timer() - t_start
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data.item())

        if words_data is not None:
            for inputs, targets in words_loader:
                elapsed = timeit.default_timer() - t_start
                optimizer.zero_grad()
                dists = calc_pair_sim(model.embed(inputs))
                loss = nn.MSELoss()(dists, targets)
                loss.backward()
                optimizer.step(lr=lr)
                epoch_words_loss.append(loss.data[0])

        emb = None
        if epoch == (opt.epochs - 1) or epoch % opt.eval_each == (opt.eval_each - 1):
            emb = model
        log.info(
            'info: {'
            f'"elapsed": {elapsed}, '
            f'"loss": {np.mean(epoch_loss)}, '
            f'"words_loss": {np.mean(epoch_words_loss) if len(epoch_words_loss) else None}'
            '}'
        )
        handler.handle(msg=(epoch, elapsed, np.mean(epoch_loss), emb))
        gc.collect()


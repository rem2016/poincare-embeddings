#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch as th
from model import WordsDataset
import os
import numpy as np
import time
import logging
import argparse
from torch.autograd import Variable
from collections import defaultdict as ddict
import torch.multiprocessing as mp
import model, train, rsgd
import join_word2vec
from data import slurp
from rsgd import RiemannianSGD
from sklearn.metrics import average_precision_score
import gc
import sys


def ranking(types, model, distfn):
    lt = th.from_numpy(model.embedding())
    embedding = Variable(lt, volatile=True)
    ranks = []
    ap_scores = []
    for s, s_types in types.items():
        s_e = Variable(lt[s].expand_as(embedding), volatile=True)
        _dists = model.dist()(s_e, embedding).data.cpu().numpy().flatten()
        _dists[s] = 1e+12
        _labels = np.zeros(embedding.size(0))
        _dists_masked = _dists.copy()
        _ranks = []
        for o in s_types:
            _dists_masked[o] = np.Inf
            _labels[o] = 1
        ap_scores.append(average_precision_score(_labels, -_dists))
        for o in s_types:
            d = _dists_masked.copy()
            d[o] = _dists[o]
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
    return np.mean(ranks), np.mean(ap_scores)


def control(queue, log, types, data, fout, distfn, nepochs, processes):
    min_rank = (np.Inf, -1)
    max_map = (0, -1)
    while True:
        gc.collect()
        msg = queue.get()
        if msg is None:
            for p in processes:
                p.terminate()
            break
        else:
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
            if mrank < min_rank[0]:
                min_rank = (mrank, epoch)
            if mAP > max_map[0]:
                max_map = (mAP, epoch)
            log.info(
                ('eval: {'
                 '"epoch": %d, '
                 '"elapsed": %.2f, '
                 '"loss": %.3f, '
                 '"mean_rank": %.2f, '
                 '"mAP": %.4f, '
                 '"best_rank": %.2f, '
                 '"best_mAP": %.4f}') % (
                     epoch, elapsed, loss, mrank, mAP, min_rank[0], max_map[0])
            )
        else:
            log.info(f'json_log: {{"epoch": {epoch}, "loss": {loss}, "elapsed": {elapsed}}}')
        if epoch >= nepochs - 1:
            log.info(
                ('results: {'
                 '"mAP": %g, '
                 '"mAP epoch": %d, '
                 '"mean rank": %g, '
                 '"mean rank epoch": %d'
                 '}') % (
                     max_map[0], max_map[1], min_rank[0], min_rank[1])
            )
            break


def setup_log(opt):
    if opt.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    log_file = f'{opt.fout}/log.log'
    log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s (%(lineno)d) %(message)s',
                                      datefmt='%H:%M:%S')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(log_level)

    log = logging.getLogger('poincare-nips17')
    log.setLevel(log_level)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    return log


def set_up_output_file_name(opt):
    if opt.w2v_nn:
        opt.fout += '_w2vnn'
    if opt.symmetrize:
        opt.fout += '_sym'
    opt.fout = f'{opt.fout}.lr={opt.lr}.dim={opt.dim}.negs={opt.negs}.burnin={opt.burnin}.batch={opt.batchsize}'
    if os.path.exists(opt.fout):
        raise EnvironmentError('There is already a output file called ' + opt.fout)
    os.makedirs(opt.fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Poincare Embeddings')
    parser.add_argument('-dim', help='Embedding dimension', type=int)
    parser.add_argument('-dset', help='Dataset to embed', type=str)
    parser.add_argument('-dset_test', help='Dataset to test', type=str, default='')
    parser.add_argument('-fout', help='Filename where to store model', type=str)
    parser.add_argument('-distfn', help='Distance function', type=str)
    parser.add_argument('-lr', help='Learning rate', type=float)
    parser.add_argument('-epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=50)
    parser.add_argument('-negs', help='Number of negatives', type=int, default=20)
    parser.add_argument('-nproc', help='Number of processes', type=int, default=5)
    parser.add_argument('-ndproc', help='Number of data loading processes', type=int, default=2)
    parser.add_argument('-eval_each', help='Run evaluation each n-th epoch', type=int, default=10)
    parser.add_argument('-burnin', help='Duration of burn in', type=int, default=20)
    parser.add_argument('-debug', help='Print debug output', action='store_true', default=False)
    parser.add_argument('-symmetrize', help='Use symmetrize data', action='store_true', default=False)
    parser.add_argument('-w2v_nn', help='Use word2vec NN to map', action='store_true', default=False)
    parser.add_argument('-nn_hidden_layer', help='NN hidden layer num', type=int, default=2)
    parser.add_argument('-nn_hidden_size', help='NN hidden layer num', type=int, default=200)
    parser.add_argument('-w2v_sim', help='Use word2vec sim to map', action='store_true', default=False)
    opt = parser.parse_args()

    set_up_output_file_name(opt)
    th.set_default_tensor_type('torch.FloatTensor')
    log = setup_log(opt)

    idx, objects, dwords = slurp(opt.dset, symmetrize=opt.symmetrize, load_word=opt.w2v_nn)

    # create adjacency list for evaluation
    test_idx = idx
    if opt.dset_test != '':
        test_idx, test_objects = slurp(opt.dset_test, symmetrize=False)
    adjacency = ddict(set)
    for i in range(len(test_idx)):
        s, o, _ = test_idx[i]
        adjacency[s].add(o)
    adjacency = dict(adjacency)

    # setup Riemannian gradients for distances
    opt.retraction = rsgd.euclidean_retraction
    if opt.distfn == 'poincare':
        distfn = model.PoincareDistance
        opt.rgrad = rsgd.poincare_grad
    elif opt.distfn == 'euclidean':
        distfn = model.EuclideanDistance
        opt.rgrad = rsgd.euclidean_grad
    elif opt.distfn == 'transe':
        distfn = model.TranseDistance
        opt.rgrad = rsgd.euclidean_grad
    else:
        raise ValueError(f'Unknown distance function {opt.distfn}')

    # initialize model and data
    if opt.w2v_nn:
        model, data, model_name, conf = model.SNGraphDataset.initialize_word2vec_nn(distfn, opt, idx, objects)
    else:
        model, data, model_name, conf = model.SNGraphDataset.initialize(distfn, opt, idx, objects)

    # Build config string for log
    conf = [
        ('distfn', '"{:s}"'),
        ('dim', '{:d}'),
        ('lr', '{:g}'),
        ('batchsize', '{:d}'),
        ('negs', '{:d}'),
        ('burnin', '{:d}'),
        ('eval_each', '{:d}'),
    ] + conf
    conf = ', '.join(['"{}": {}'.format(k, f).format(getattr(opt, k)) for k, f in conf])
    log.info(f'json_conf: {{{conf}}}')

    # initialize optimizer
    optimizer = RiemannianSGD(
        model.parameters(),
        rgrad=opt.rgrad,
        retraction=opt.retraction,
        lr=opt.lr,
    )

    # if nproc == 0, run single threaded, otherwise run Hogwild
    if opt.nproc == 0:
        train.train(model, data, optimizer, opt, log, 0)
    else:
        queue = mp.Manager().Queue()
        model.share_memory()
        processes = []
        for rank in range(opt.nproc):
            if opt.w2v_sim:
                word_data = WordsDataset()
                p = mp.Process(
                    target=join_word2vec.train,
                    args=(model, data, word_data, optimizer, opt, log, rank + 1, queue)
                )
            else:
                p = mp.Process(
                    target=train.train_mp,
                    args=(model, data, optimizer, opt, log, rank + 1, queue)
                )
            p.start()
            processes.append(p)

        ctrl = mp.Process(
            target=control,
            args=(queue, log, adjacency, data, opt.fout, distfn, opt.epochs, processes)
        )
        ctrl.start()
        ctrl.join()

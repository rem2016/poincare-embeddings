#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch as th
from data_loader import WordsDataset
import data_loader
import os
import numpy as np
import time
import logging
import argparse
from threading import RLock
from evaluation import Evaluator
from torch.autograd import Variable
from concurrent.futures import ThreadPoolExecutor
from word_vec_loader import WordVectorLoader
from collections import defaultdict as ddict
import torch.multiprocessing as mp
import model, train, rsgd
import join_word2vec
from data import slurp
import random
from rsgd import RiemannianSGD
from sklearn.metrics import average_precision_score
import gc
import sys


def random_sample(adj, num):
    if len(adj) <= num:
        return adj.items()
    _data = [(i, j) for i, j in adj.items()]
    return random.sample(_data, num)


def ranking(types, _model, distfn, sense_num=1000000, max_workers=3):
    MAX_NODE_NUM = 5000  # avoid too large set (need more than 3 hour for whole noun set)
    with th.no_grad():
        lt = th.from_numpy(_model.embedding())
        embedding = Variable(lt)
        ranks = []
        ap_scores = []
        lock = RLock()

        def work(_s, _s_types):
            nonlocal ranks
            s_e = Variable(lt[_s].expand_as(embedding))
            _dists = _model.dist()(s_e, embedding).data.cpu().numpy().flatten()
            _dists[_s] = 1e+12
            _labels = np.zeros(embedding.size(0))
            _dists_masked = _dists.copy()
            _ranks = []
            for o in _s_types:
                _dists_masked[o] = np.Inf
                _labels[o] = 1
            # Caution
            # ignore all words
            for i in range(sense_num, len(_dists)):
                _dists_masked[i] = np.Inf
            for o in _s_types:
                d = _dists_masked.copy()
                d[o] = _dists[o]
                r = np.argsort(d)
                _ranks.append(np.where(r == o)[0][0] + 1)
            ap = average_precision_score(_labels, -_dists)
            lock.acquire()
            ap_scores.append(ap)
            ranks += _ranks
            lock.release()

        with ThreadPoolExecutor(max_workers=max_workers) as worker:
            for s, s_types in random_sample(types, MAX_NODE_NUM):
                worker.submit(work, s, s_types)
    return np.mean(ranks), np.mean(ap_scores)


def eval_human(_model, objs, word_index=None):
    ev = Evaluator(_model.embedding(), objs, word_index=word_index)
    return ev.evaluate(try_use_word=True)


def control(queue, log, train_adj, test_adj, data, fout, distfn, nepochs, processes, w2v_nn, w2v_sim):
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
            epoch, elapsed, loss, model, word_sim_loss = msg
        if model is not None:
            # save model to fout
            _fout = f'{fout}/{epoch}.nth'
            log.info(f'Saving model f{_fout}')
            log.info(str(eval_human(model, data.objects, WordVectorLoader.index2word)))
            th.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'objects': data.objects,
                'word_index': WordVectorLoader.index2word
            }, _fout)

            # TODO if any error occurs here, refactor
            # compute embedding quality
            log.info('Computing ranking')
            _start_time = time.time()
            train_mrank, train_mAP = ranking(train_adj, model, distfn, len(data.objects))
            mrank, mAP = train_mrank, train_mAP
            test_info = ''
            if test_adj is not None:
                test_mrank, test_mAP = ranking(test_adj, model, distfn, len(data.objects))
                mrank, mAP = test_mrank, test_mAP
                test_info = f', test_mean_rank: {test_mrank}, test_mAP: {test_mAP}, word_sim_loss: {word_sim_loss}'
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
                 '"train_mean_rank": %.2f, '
                 '"train_mAP": %.4f, '
                 '"best_rank": %.2f, '
                 '"best_mAP": %.4f%s}') % (
                    epoch, elapsed, loss, train_mrank, train_mAP, min_rank[0], max_map[0], test_info)
            )
        else:
            log.info(f'json_log: {{"epoch": {epoch}, "loss": {loss}, '
                     f'"words_sim_loss": {word_sim_loss}, "elapsed": {elapsed}}}')
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
    if opt.w2v_sim:
        opt.fout += '_w2vsim'
    if opt.symmetrize:
        opt.fout += '_sym'
    opt.fout = f'{opt.fout}.lr={opt.lr}.dim={opt.dim}.negs={opt.negs}.burnin={opt.burnin}.batch={opt.batchsize}'
    if os.path.exists(opt.fout):
        if opt.override:
            print("This will rename the original result. Make sure you have closed the corresponding program [Y/n]")
            s = 'y'
            if s.lower() == 'n':
                sys.exit(0)
            else:
                i = 0
                while os.path.exists(f'{opt.fout}[{i}]'):
                    i += 1
                os.rename(opt.fout, f'{opt.fout}[{i}]')
        else:
            print("This will rename the original result. Make sure you have closed the corresponding program [y/N]")
            s = input()
            if s.lower() != 'y':
                sys.exit(0)
            else:
                i = 0
                while os.path.exists(f'{opt.fout}[{i}]'):
                    i += 1
                os.rename(opt.fout, f'{opt.fout}[{i}]')

    os.makedirs(opt.fout)


def get_adjacency_by_idx(_idx):
    _adjacency = ddict(set)
    for i in range(len(_idx)):
        s, o, _ = _idx[i]
        _adjacency[s].add(o)
    _adjacency = dict(_adjacency)
    return _adjacency


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
    parser.add_argument('-nn_hidden_layer', help='NN hidden layer num', type=int, default=1)
    parser.add_argument('-nn_hidden_size', help='NN hidden layer num', type=int, default=50)
    parser.add_argument('-w2v_sim', help='Use word2vec sim to map', action='store_true', default=False)
    parser.add_argument('-word', help='Link words to data', action='store_true', default=False)
    parser.add_argument('-override', help='Override result with the same name', action='store_true', default=False)
    opt = parser.parse_args()

    set_up_output_file_name(opt)
    th.set_default_tensor_type('torch.FloatTensor')
    log = setup_log(opt)
    log.info("======================================")

    log.info("Config")
    log.info(str(opt))

    idx, objects, dwords = slurp(opt.dset, symmetrize=opt.symmetrize,
                                 load_word=opt.w2v_nn or opt.w2v_sim or opt.word,
                                 build_word_vector=True)

    # create adjacency list for evaluation
    test_adjacency = None
    train_adjacency = get_adjacency_by_idx(idx)
    if opt.dset_test != '':
        test_idx, test_objects, test_dwords = slurp(opt.dset_test,
                                                    symmetrize=opt.symmetrize,
                                                    load_word=False,  # Test test set should be the same
                                                    build_word_vector=False,
                                                    objects=objects)
        test_adjacency = get_adjacency_by_idx(test_idx)

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
    word_as_head_data = None
    word_as_neg_data = None
    if opt.w2v_nn:
        model, data, model_name, conf = data_loader.SNGraphDataset.initialize_word2vec_nn(distfn, opt, idx, objects)
        word_as_head_data = data_loader.WordAsHeadDataset(idx, objects, opt.negs, sense_num=len(objects))
        word_as_neg_data = data_loader.WordAsNegDataset(idx, objects, opt.negs, words_num=len(dwords))
    else:
        num = len(objects) + (len(dwords) if dwords is not None else 0)
        model, data, model_name, conf = data_loader.SNGraphDataset.initialize(distfn, opt, idx, objects, node_num=num)

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

    if opt.nproc == 0:
        handler = train.SingleThreadHandler(log, train_adjacency, test_adjacency, data, opt.fout, distfn, ranking)
        if opt.w2v_sim:
            train.single_thread_train(model, data, optimizer, opt, log,
                                      handler,
                                      words_data=WordsDataset(WordVectorLoader.word_vec),
                                      w_head_data=word_as_head_data,
                                      w_neg_data=word_as_neg_data)
        else:
            train.single_thread_train(model, data, optimizer, opt, log, handler,
                                      w_head_data=word_as_head_data,
                                      w_neg_data=word_as_neg_data)
    else:
        queue = mp.Manager().Queue()
        model.share_memory()
        processes = []
        for rank in range(opt.nproc):
            if opt.w2v_sim:
                word_data = WordsDataset(WordVectorLoader.word_vec, sense_num=len(objects))
                p = mp.Process(
                    target=join_word2vec.combine_w2v_sim_train,
                    args=(model, data, word_data, optimizer, opt, log, rank + 1, queue)
                )
            else:
                p = mp.Process(
                    target=train.train_mp,
                    args=(model, data, optimizer, opt, log, rank + 1,
                          queue, word_as_head_data, word_as_neg_data)
                )
            p.start()
            processes.append(p)

        ctrl = mp.Process(
            target=control,
            args=(
                queue, log, train_adjacency, test_adjacency, data, opt.fout,
                distfn, opt.epochs, processes, opt.w2v_nn, opt.w2v_sim)
        )
        ctrl.start()
        ctrl.join()

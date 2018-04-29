import torch as th
from model import WordsDataset
import logging
from collections import defaultdict as ddict
import torch.multiprocessing as mp
import join_word2vec
from data import slurp
from rsgd import RiemannianSGD
import train, rsgd
import multiprocessing
import model
from argparse import Namespace
from embed import ranking


def control(model, log, types, data, fout, distfn, nepochs, processes, w2v_nn, w2v_sim):
    mrank, mAP = ranking(types, model, distfn)


def test_rank():
    opt = Namespace(batchsize=10, burnin=10, debug=False,
               dim=5, distfn='poincare', dset='wordnet/debug.tsv',
               dset_test='', epochs=300, eval_each=1,
               fout='model/zzzzTest_w2vnn.lr=0.3.dim=5.negs=50.burnin=10.batch=10',
               lr=0.3, ndproc=2, negs=50, nn_hidden_layer=1,
               nn_hidden_size=100, nproc=1, override=True,
               symmetrize=False, w2v_nn=True, w2v_sim=False)

    th.set_default_tensor_type('torch.FloatTensor')
    log = logging.getLogger("Test")
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
        _model, data, model_name, conf = model.SNGraphDataset.initialize_word2vec_nn(distfn, opt, idx, objects)
    else:
        _model, data, model_name, conf = model.SNGraphDataset.initialize(distfn, opt, idx, objects)

    mrank, mAP = ranking(adjacency, _model, distfn)


import numpy as np
import spacy
import torch
from embed import ranking, get_adjacency_by_idx, eval_human
from nltk.corpus import wordnet
import eval_scws
from model import PoincareDistance
import time
import torch as th
from data import slurp
import os
from IPython.display import display
from model import SNEmbedding
import evaluation
import json
import re
import pickle
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class LogInfo:
    def __init__(self, config: dict, loss: list, ev: list, ks: list):
        self.config = config
        self.loss = loss
        self.eval = ev
        self.validate()
        self._ks = ks

    def validate(self):
        pass

    def get_word_loss(self):
        return [d['word_sim_loss'] for d in self.loss]

    def get_loss(self):
        return [d['loss'] for d in self.loss]

    def get_eval_result(self):
        ddict = defaultdict(lambda: [])
        for d in self.eval:
            for k, v in d.items():
                ddict[k].append(v)
        return pd.DataFrame(dict(ddict))

    @property
    def k(self):
        return self._ks[-1]

    @property
    def name(self):
        return ''

    @classmethod
    def load_data(cls, file_path):
        loss = []
        config = {}
        ev = []
        ks = [1]
        with open(file_path) as f:
            for line in f:
                try:
                    if 'json_log' in line:
                        ans = re.findall(r'json_log: ({.*?})', line)[0].replace('None', 'null').strip()
                        loss.append(json.loads(ans))
                    elif 'eval:' in line:
                        ans = re.findall(r'eval: ({.*?})', line)[0].replace('None', 'null').strip()
                        ans = re.sub(r'\s(\w+):', r'"\1":', ans)
                        ev.append(json.loads(ans))
                    elif 'Namespace' in line:
                        ans = re.findall(r'Namespace(\(.*?\))', line)[0]
                        config = eval('dict' + ans)
                    elif 'k_param=' in line:
                        ans = re.findall(r'k_param=([\d\.]+)', line)[0]
                        ks.append(float(ans))
                except Exception as e:
                    print(line, ans)
                    raise e
        return cls(config, loss, ev, ks)


class ModelInfo:
    def __init__(self, path_to_dir: str):
        self.log = LogInfo.load_data(os.path.join(path_to_dir, 'log.log'))
        self.path_to_dir = path_to_dir
        self.model_paths = [(os.path.join(path_to_dir, x), int(x.split('.')[0])) for x in os.listdir(path_to_dir) if
                            x.endswith('.nth')]
        self.model_paths.sort(key=lambda x: x[1])
        self.epochs = [x[1] for x in self.model_paths]
        self.model_paths = [x[0] for x in self.model_paths]
        self.name = self.load_name()

    def load_name(self):
        config = self.log.config
        name = [f"{config.get('dim', 0)}d"]
        if config.get('word', False):
            name.append('Word')
        if config.get('w2v_sim', False):
            name.append('Sim')
        if config.get('w2v_nn', False):
            name.append('NN')
        if self.log.config.get('distfn', 'poincare') != 'poincare':
            name.append(self.log.config['distfn'])
        return '-'.join(name)

    def load_k_models(self, k=3):
        n = len(self.model_paths)
        chosen_models = [self.model_paths[i] for i in range(0, n, n // k)]
        epochs = [self.epochs[i] for i in range(0, n, n // k)]
        return zip(epochs, chosen_models)

    def path_to_the_best_model(self):
        return \
            max([(
                int(x.split('.')[0]),
                os.path.join(self.path_to_dir, x)
            ) for x in os.listdir(self.path_to_dir) if x.endswith('.nth')],
                key=lambda x: x[0])[1]

    def load_model(self):
        """

        :return: model_, objs, index2word
        """
        model_ = th.load(self.path_to_the_best_model())
        objs = model_['objects']
        index2word = model_['word_index']
        emb = model_['model']
        model_ = _load_model(emb, emb['lt.weight'].size(0), emb['lt.weight'].size(1))
        return model_, objs, index2word, emb['lt.weight']


def is_not_important(dir_path, threshold=200):
    for file in os.listdir(dir_path):
        try:
            num = int(file.split('.')[0])
        except ValueError:
            continue
        if num >= threshold:
            return False
    return True


def load_dir(dir_path):
    info = LogInfo.load_data(os.path.join(dir_path, 'log.log'))
    return info


def load_all_important_models(root='..\\model\\data\\', threshold=200, is_module_dir=False):
    data = {}
    for models_dir in os.listdir(root):
        if models_dir[0] == '.':
            continue
        models_dir = os.path.join(root, models_dir)
        if not is_module_dir:
            for model_dir in os.listdir(models_dir):
                model_dir = os.path.join(models_dir, model_dir)
                if is_not_important(model_dir, threshold):
                    continue
                data[model_dir] = ModelInfo(model_dir)
        else:
            if is_not_important(models_dir, threshold):
                continue
            data[models_dir] = ModelInfo(models_dir)
    return data


def load_all_important_log(root='..\\model\\data\\'):
    data = {}
    for models in os.listdir(root):
        if models[0] == '.':
            continue
        models = root + models
        for model in os.listdir(models):
            model = os.path.join(models, model)
            if is_not_important(model):
                continue
            data[model] = load_dir(model)
    return data


def reeval_rank():
    models = load_all_important_models()
    print('Total model num', len(models))
    result = {}
    for name, model in models.items():
        print('=====================================')
        print(f'{name}')
        print(f'{model.log.config}')
        print(f'{model.name}')
        ranks = []
        maps = []
        for epoch, mino in model.load_k_models(4):
            ev = evaluation.Evaluator.initialize_by_file(mino)
            rank, mAP = ev.rank()
            print('{:<5}{:<33}{:<33}'.format(epoch, rank, mAP))
            ranks.append(rank)
            maps.append(mAP)
        result[name] = {'rank': ranks, 'maps': maps}
        print()
    display(result)
    with open('./experiment/ans.json', 'w') as f:
        json.dump(result, f, indent=2)


def reeval_sim():
    models = load_all_important_models()
    print('Total model num', len(models))
    result = {}
    for name, model in models.items():
        print('=====================================')
        print(f'{name}')
        print(f'{model.log.config}')
        print(f'{model.name}')
        ranks = []
        maps = []
        for epoch, mino in model.load_k_models(4):
            ev = evaluation.Evaluator.initialize_by_file(mino)
            for t in ('tanh', 'neg', 'reciprocal', 'exp'):
                corr = ev.evaluate(method=t, try_use_word=True)
                print('{:<5}{:<10}{:<93}'.format(epoch, t, corr))
            ranks.append(rank)
            maps.append(mAP)
        result[name] = {'rank': ranks, 'maps': maps}
        print()
    display(result)
    with open('./experiment/ans.json', 'w') as f:
        json.dump(result, f, indent=2)


def _load_model(state_dict, size, dim):
    _model = SNEmbedding(size, dim)
    _model.load_state_dict(state_dict)
    return _model


def re_rank(model_info, train_set, test_set):
    model_ = th.load(model_info.path_to_the_best_model())
    objs = model_['objects']
    index2word = model_['word_index']
    emb = model_['model']
    model_ = _load_model(emb, emb['lt.weight'].size(0), emb['lt.weight'].size(1))
    idx, objects, dwords = slurp(train_set,
                                 load_word=model_info.log.config['w2v_sim'] or model_info.log.config['word'],
                                 build_word_vector=True,
                                 objects=objs)
    train_adjacency = get_adjacency_by_idx(idx)
    test_idx, test_objects, test_dwords = slurp(test_set,
                                                load_word=False,  # Test test set should be the same
                                                build_word_vector=False,
                                                objects=objects)
    test_adjacency = get_adjacency_by_idx(test_idx)

    ans = {}
    print("Start ranking train set")
    ans['train'] = ranking(train_adjacency, model_, None, len(objects), mask_types=test_adjacency, max_workers=1)
    print("Start ranking test set")
    ans['test'] = ranking(test_adjacency, model_, None, len(objects), mask_types=train_adjacency, max_workers=1)
    return ans


def re_eval_sim(model_info):
    model_ = th.load(model_info.path_to_the_best_model())
    objs = model_['objects']
    index2word = model_['word_index']
    emb = model_['model']
    model_ = _load_model(emb, emb['lt.weight'].size(0), emb['lt.weight'].size(1))
    ans = {}
    ans['Synset'] = eval_human(model_,
                               objs,
                               index2word,
                               method='reciprocal',
                               use_word=False)
    ans['Word Cos'] = eval_human(model_,
                                 objs,
                                 index2word,
                                 use_word=True,
                                 method='cos')
    ans['Word Rec']=eval_human(model_,
                               objs,
                               index2word,
                               use_word=True,
                               method='reciprocal')
    return ans


def eval_on_scws(model_info):
    class NotEmbeddingError(ValueError):
        pass

    def _load_scws_data():
        _data = []
        with open('../SCWS/ratings.txt') as f:
            for line in f:
                _, a, pos, b, pos_, s1, s2, score, *args = line.split('\t')
                if pos != 'n' or pos_ != 'n':
                    continue
                _data.append(eval_scws.Td(a, b, s1, s2, score))
        return _data

    def _calc_sim(td, use_idf=False):
        try:
            l_emb = _most_sim_synset_embedding(td.get_l_context(), td.l_word, use_idf=use_idf)
            r_emb = _most_sim_synset_embedding(td.get_r_context(), td.r_word, use_idf=use_idf)
            return -PoincareDistance()(l_emb, r_emb)
        except NotEmbeddingError:
            return None

    def _calc_sim_closest(td):
        l_synsets = [x.name() for x in wordnet.synsets(td.l_word) if x.name() in iobjs]
        r_synsets = [x.name() for x in wordnet.synsets(td.r_word) if x.name() in iobjs]
        if len(l_synsets) == 0 or len(r_synsets) == 0:
            return None
        return max([_neg_sim(s1, s2) for s1 in l_synsets for s2 in r_synsets])

    def _cos_sim(v1, v2):
        def norm(v):
            return th.sqrt(th.sum(v**2, -1))

        ret = th.sum(v1 * v2, dim=-1) / (norm(v1) * norm(v2))
        return ret.squeeze(-1)

    def _neg_sim(s1, s2):
        v1 = emb[iobjs[s1]]
        v2 = emb[iobjs[s2]]
        dist = PoincareDistance()(v1, v2)
        return -dist

    def _most_sim_synset_embedding(context, _word, use_idf=False):
        nonlocal emb, objs
        context += [_word] * 3
        _context = _sum_context(context, use_idf=use_idf)
        max_emb = None
        max_sim = -1
        for synset in wordnet.synsets(_word):
            if synset.name() in iobjs:
                _emb = emb[iobjs[synset.name()]]
                cos_sim = _cos_sim(_context, _emb)
                if cos_sim > max_sim:
                    max_sim = cos_sim
                    max_emb = _emb
        if max_emb is None:
            raise NotEmbeddingError()
        return max_emb

    def _sum_context(context, use_idf=False):
        _ans = None
        for word in context:
            if word not in word2index and word.lower() in word2index:
                word = word.lower()
            if word in word2index:
                weight = 1
                if use_idf:
                    weight = tfidf.idf(word)
                if _ans is None:
                    _ans = emb[word2index[word]] * weight
                else:
                    _ans += emb[word2index[word]] * weight
        if _ans is None:
            raise ValueError("cannot find any context word in the embedding result")
        return _ans

    def _calc_sim_word_cos(td):
        if td.l_word not in word2index or td.r_word not in word2index:
            return None
        return _cos_sim(emb[word2index[td.l_word]], emb[word2index[td.r_word]])

    def _calc_sim_word_dist(td):
        if td.l_word not in word2index or td.r_word not in word2index:
            return None
        return -PoincareDistance()(emb[word2index[td.l_word]], emb[word2index[td.r_word]])

    def _eval(func):
        try:
            sims = [func(t) for t in scws]
            human = [t.score for t in scws]
            sims, human = zip(*[(float(x), float(y)) for x, y in zip(sims, human) if x is not None])
            print(f"Sims num {len(sims)}/{len(scws)}")
            return np.corrcoef(sims, human)[0, 1]
        except TypeError:
            return None
        except Exception as e:
            raise e

    def _word2vec_sim(td):
        a = nlp(td.l_word)
        b = nlp(td.r_word)
        va = a.vector
        vb = b.vector
        if np.sum(np.abs(va)) < 1e-6 or np.sum(np.abs(vb)) < 1e-6:
            return None
        return _cos_sim(torch.Tensor(va), torch.Tensor(vb))

    start = time.time()
    scws = _load_scws_data()
    print(time.time() - start)
    tfidf = eval_scws.Tfidf()
    tfidf.fit()

    nlp = spacy.load('en_core_web_lg')
    model_, objs, index2word, emb = model_info.load_model()
    iobjs = {name: i for i, name in enumerate(objs)}
    word2index = {name: i for i, name in enumerate(index2word)} if index2word is not None else None
    print("Model loading finished. Start calculating")
    ans = {}
    ans['no_idf'] = _eval(_calc_sim)
    ans['with_idf'] = _eval(lambda t: _calc_sim(t, use_idf=True))
    ans['closest'] = _eval(_calc_sim_closest)
    ans['word_cos'] = _eval(_calc_sim_word_cos)
    ans['word_dist'] = _eval(_calc_sim_word_dist)
    # ans['word2vec'] = _eval(_word2vec_sim)  # 0.64454779 1323/1328
    return ans


def re_rank_model(path, name=''):
    print('=============================================')
    print(f'\n      Start: {name}\n')
    print('=============================================')
    print(path)
    start = time.time()
    model_info = ModelInfo(path)
    train = model_info.log.config['dset']
    test = model_info.log.config['dset_test']
    print(re_rank(model_info, train, test))
    print('Used time', time.time() - start)
    print('======================================')


def re_eval_model_sim(path, name=''):
    print('=============================================')
    print(f'\n      Start: {name}\n')
    print('=============================================')
    print(path)
    start = time.time()
    model_info = ModelInfo(path)
    print(re_eval_sim(model_info))
    print('Used time', time.time() - start)
    print('======================================')


def re_eval_model_scws(path, name=''):
    print('=============================================')
    print(f'\n      Start: {name}\n')
    print('=============================================')
    print(path)
    start = time.time()
    model_info = ModelInfo(path)
    print(eval_on_scws(model_info))
    print('Used time', time.time() - start)
    print('======================================')


if __name__ == '__main__':
    models = load_all_important_models()
    for _model in models:
        try:
            re_eval_model_scws(_model)
        except:
            pass
    # re_eval_model_scws(r'C:\Users\Administrator\Documents\G\model\data\LatestHuge'
    #                   r'\GRADUALLY.50d.cos.train_w2vsim_cos_imb.lr=1.0.dim=50.negs=50.burnin=20.batch=50',
    #                   'Gradually burnin')
    # re_eval_model_scws(r'C:\Users\Administrator\Documents\G\model\data\LatestMid'
    #                   r'\BEST.nouns.50d.cos.train_w2vsim_cos_imb.lr=1.0.dim=50.negs=50.burnin=20.batch=50',
    #                   'Default 50d cos')
    # re_eval_model_scws(r'C:\Users\Administrator\Documents\G\model\data\LatestSm'
    #                   r'\nouns.50d.train.lr=1.0.dim=50.negs=50.burnin=20.batch=50',
    #                   'Naive 50d')

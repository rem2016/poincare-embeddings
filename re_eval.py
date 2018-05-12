import numpy as np
from sklearn.linear_model import LinearRegression
import spacy
from sematch.evaluation import WordSimDataset
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

word_sim_data = WordSimDataset()
refer_pairs, refer_human_ground_truth = word_sim_data.load_dataset('noun_rg')
refer_pairs = list(refer_pairs)
datasets = ['noun_rg', 'noun_mc', 'noun_ws353', 'noun_ws353-sim', 'noun_simlex']
ground_truth_pairs = [word_sim_data.load_dataset(dataset) for dataset in datasets]
nlp = spacy.load('en_core_web_lg')


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
        index2word = model_.get('word_index', None)
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
    for name, model in models.items():
        print('=====================================')
        print(f'{name}')
        print(f'{model.log.config}')
        print(f'{model.name}')
        for epoch, mino in model.load_k_models(4):
            ev = evaluation.Evaluator.initialize_by_file(mino)
            for t in ('tanh', 'neg', 'reciprocal', 'exp'):
                corr = ev.evaluate(method=t, try_use_word=True)
                print('{:<5}{:<10}{:<93}'.format(epoch, t, corr))
        print()


def _load_model(state_dict, size, dim):
    _model = SNEmbedding(size, dim)
    if 'k' in state_dict:
        del state_dict['k']
        del state_dict['b']
    _model.load_state_dict(state_dict)
    return _model


def re_rank(model_info, train_set, test_set):
    model_ = th.load(model_info.path_to_the_best_model())
    objs = model_['objects']
    index2word = model_.get('word_index', None)
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

        _norm = (norm(v1) * norm(v2))
        if _norm < 1e-10:
            return None
        ret = th.sum(v1 * v2, dim=-1) / _norm
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
            raise NotEmbeddingError("cannot find any context word in the embedding result")
        return _ans

    def _calc_sim_word_cos(td):
        if word2index is None or td.l_word not in word2index or td.r_word not in word2index:
            return None
        return _cos_sim(emb[word2index[td.l_word]], emb[word2index[td.r_word]])

    def _calc_sim_word_dist(td):
        if word2index is None or td.l_word not in word2index or td.r_word not in word2index:
            return None
        return -PoincareDistance()(emb[word2index[td.l_word]], emb[word2index[td.r_word]])

    def _eval(func):
        try:
            sims = [func(t) for t in scws]
            human = [t.score for t in scws]
            sims, human = zip(*[(float(x), float(y)) for x, y in zip(sims, human) if x is not None])
            print(f"        Sims num {len(sims)}/{len(scws)}")
            return np.corrcoef(sims, human)[0, 1]
        except (TypeError, ValueError):
            return None
        except Exception as e:
            raise e

    def _fit_simlex(*sim_funcs):
        tds = [eval_scws.Td(x, y, x, y, s) for (x, y), s in zip(refer_pairs, refer_human_ground_truth)]
        _preds = [[sim_func(td) for sim_func in sim_funcs] for td in tds]
        _X = []
        _y = []
        for _pred, score in zip(_preds, refer_human_ground_truth):
            if any((x is None for x in _pred)):
                continue
            _X.append([float(x) for x in _pred])
            _y.append(score)
        if len(_X) == 0:
            return None
        _lr = LinearRegression()
        _lr.fit(_X, _y)
        print(f'Fit simlex, final corr with simlex = {np.corrcoef(_lr.predict(_X), _y)[0, 1]}')
        return _lr

    def _calc_sim_lr_simlex(td):
        a = _calc_sim_closest(td)
        b = _calc_sim_word_cos(td)
        return a is None or b is None or lr is None or lr.predict([[a, b]])

    def _calc_sim_lr_simlex_context(td):
        a = _calc_sim(td)
        b = _calc_sim_word_cos(td)
        return a is None or b is None or lr is None or lr.predict([[a, b]])

    def _word2vec_sim(td):
        a = nlp(td.l_word)
        b = nlp(td.r_word)
        va = a.vector
        vb = b.vector
        if np.sum(np.abs(va)) < 1e-6 or np.sum(np.abs(vb)) < 1e-6:
            return None
        return _cos_sim(torch.Tensor(va), torch.Tensor(vb))

    def _calc_all_sim_lr():
        if lr is None:
            return None

        def __eval(_p, _hs):
            tds = [eval_scws.Td(x, y, x, y, s) for (x, y), s in zip(_p, _hs)]
            X = [[_calc_sim_closest(td), _calc_sim_word_cos(td)] for td in tds]
            x1s = [x[1] for x in X if x[1] is not None]
            mean_x1 = sum(x1s) / len(x1s)
            x0s = [x[0] for x in X if x[0] is not None]
            mean_x0 = sum(x0s) / len(x0s)
            pred = []
            y = []
            for _x, _h in zip(X, _hs):
                if _x[0] is None and _x[1] is None:
                    continue
                y.append(float(_h))
                if _x[1] is None:
                    _x[1] = mean_x1
                elif _x[0] is None:
                    _x[0] = mean_x0
                try:
                    pred.append(float(lr.predict([_x])[0]))
                except ValueError:
                    print(_x)
            if len(pred) == 0:
                return None
            return np.corrcoef(pred, y)[0, 1]

        _ret = {}
        for name, data in zip(datasets, ground_truth_pairs):
            pairs, human = data
            _ret[name] = __eval(pairs, human)
        return _ret

    def _calc_closest_sim():
        def __eval(_p, _hs):
            tds = [eval_scws.Td(x, y, x, y, s) for (x, y), s in zip(_p, _hs)]
            X = [_calc_sim_closest(td) for td in tds]
            y = [h for _x, h in zip(X, _hs) if _x is not None]
            X = [_x for _x in X if _x is not None]
            if len(X) == 0:
                return None
            return np.corrcoef(X, y)[0, 1]

        _ret = {}
        for name, data in zip(datasets, ground_truth_pairs):
            pairs, human = data
            _ret[name] = __eval(pairs, human)
        return _ret

    start = time.time()
    scws = _load_scws_data()
    tfidf = eval_scws.Tfidf()
    tfidf.fit()

    model_, objs, index2word, emb = model_info.load_model()
    iobjs = {name: i for i, name in enumerate(objs)}
    word2index = {name: i for i, name in enumerate(index2word)} if index2word is not None else None
    lr = _fit_simlex(_calc_sim_closest, _calc_sim_word_cos)
    print("        Model loading finished. Start calculating")
    ans = {}
    ans['no_idf_context'] = _eval(_calc_sim)
    ans['with_idf_context'] = _eval(lambda t: _calc_sim(t, use_idf=True))
    ans['closest'] = _eval(_calc_sim_closest)
    ans['word_cos'] = _eval(_calc_sim_word_cos)
    ans['word_dist'] = _eval(_calc_sim_word_dist)
    ans['lr_closest'] = _eval(_calc_sim_lr_simlex)
    ans['lr_context'] = _eval(_calc_sim_lr_simlex_context)
    ans['closest_sim'] = _calc_closest_sim()
    ans['all_sim_lr'] = _calc_all_sim_lr()
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
    print(model_info.path_to_the_best_model().split('\\')[-1])
    for k, v in eval_on_scws(model_info).items():
        print('- {: <22}{: <22}'.format(str(k), str(v)))
    print('Used time', time.time() - start)
    print('======================================\n\n\n')


if __name__ == '__main__':
    re_rank_model(r"C:\Users\Administrator\Documents\G\model\data\PURE10d")
    re_eval_model_scws(r"C:\Users\Administrator\Documents\G\model\data\PURE10d")
    # re_rank_model(r'C:\Users\Administrator\Documents\G\model\data\80D512\80D_w2vsim_imb.lr=1.0.dim=80.negs=50.burnin=20.batch=50')
    # re_eval_model_scws(r'C:\Users\Administrator\Documents\G\model\data\80D512\80D_w2vsim_imb.lr=1.0.dim=80.negs=50.burnin=20.batch=50')
    # models = load_all_important_models()
    # for _model in models:
    #     try:
    #         re_eval_model_scws(_model)
    #     except Exception as e:
    #         raise e
    # re_eval_model_scws(r'C:\Users\Administrator\Documents\G\model\data\LatestHuge'
    #                   r'\GRADUALLY.50d.cos.train_w2vsim_cos_imb.lr=1.0.dim=50.negs=50.burnin=20.batch=50',
    #                   'Gradually burnin')
    # re_eval_model_scws(r'C:\Users\Administrator\Documents\G\model\data\LatestMid'
    #                   r'\BEST.nouns.50d.cos.train_w2vsim_cos_imb.lr=1.0.dim=50.negs=50.burnin=20.batch=50',
    #                   'Default 50d cos')
    # re_eval_model_scws(r'C:\Users\Administrator\Documents\G\model\data\LatestSm'
    #                   r'\nouns.50d.train.lr=1.0.dim=50.negs=50.burnin=20.batch=50',
    #                   'Naive 50d')


r"""

=============================================

      Start: Pure 10d

=============================================
C:\Users\Administrator\Documents\G\model\data\PURE10d
Loading wordnet words
Total 37539 words are added
Loaded all_nn.tsv. Words:  37539, num: 578053, average: 15.398731985401849
slurp: objects=82115, edges=767086, words=37539

Start ranking train set
Start ranking test set
{'train': (6.23669568632585, 0.5353748937148014), 'test': (5.518334745249909, 0.5210224501164599)}
Used time 523.3364696502686
======================================
=============================================

      Start: Pure 10d

=============================================
C:\Users\Administrator\Documents\G\model\data\PURE10d
824.nth
        Model loading finished. Start calculating
        Sims num 1299/1328
        Sims num 1328/1328
C:\Users\Administrator\Anaconda3\lib\site-packages\numpy\lib\function_base.py:3183: RuntimeWarning: invalid value encountered in true_divide
  c /= stddev[:, None]
C:\Users\Administrator\Anaconda3\lib\site-packages\numpy\lib\function_base.py:3184: RuntimeWarning: invalid value encountered in true_divide
  c /= stddev[None, :]
- no_idf_context        None                  
- with_idf_context      None                  
- closest               0.45485046032399784   
- word_cos              None                  
- word_dist             None                  
- lr_closest            nan                   
- lr_context            None                  
- closest_sim           {'noun_rg': 0.7950109570207747, 'noun_mc': 0.7856466620769119, 'noun_ws353': 0.30704477983325423, 'noun_ws353-sim': 0.5810733842950273, 'noun_simlex': 0.5544896524660374}
- all_sim_lr            None                  
Used time 6.792054653167725
======================================


=============================================

      Start: 80d gradually

=============================================
C:\Users\Administrator\Documents\G\model\data\80D512\80D_w2vsim_imb.lr=1.0.dim=80.negs=50.burnin=20.batch=50
Loading wordnet words
Total 37422 words are added
Loaded all_nn.tsv. Words:  37422, num: 575073, average: 15.367243867243868
slurp: objects=81690, edges=766765, words=37422
[ERROR]: Total 24 nodes weren't found in the train set

Start ranking train set
Start ranking test set
{'train': (27.496974822718105, 0.5796454589927496), 'test': (27.889952733002062, 0.5290506835705141)}
Used time 1076.9975526332855
======================================
=============================================

      Start: 80d gradually

=============================================
C:\Users\Administrator\Documents\G\model\data\80D512\80D_w2vsim_imb.lr=1.0.dim=80.negs=50.burnin=20.batch=50
199.nth
Fit simlex, final corr with simlex = 0.7694646909933712
        Model loading finished. Start calculating
        Sims num 1298/1328
        Sims num 1298/1328
        Sims num 1298/1328
        Sims num 1256/1328
        Sims num 1256/1328
        Sims num 1328/1328
        Sims num 1328/1328
- no_idf_context        0.380920090228718     
- with_idf_context      0.37051007360327554   
- closest               0.39002089875828544   
- word_cos              0.3758685372651869    
- word_dist             0.44063017797257786   
- lr_closest            0.427079433517945     
- lr_context            0.42279512829325977   
- closest_sim           {'noun_rg': 0.7031826226367452, 'noun_mc': 0.6867344709022339, 'noun_ws353': 0.24921007970810885, 'noun_ws353-sim': 0.5076110117558662, 'noun_simlex': 0.4612737021659282}
- all_sim_lr            {'noun_rg': 0.7694646909933712, 'noun_mc': 0.7471274785884376, 'noun_ws353': 0.2782686730163062, 'noun_ws353-sim': 0.5790531576930787, 'noun_simlex': 0.48519162734707605}
Used time 10.489257574081421
======================================

"""
import numpy as np

import os
from IPython.display import display
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


def load_all_important_log(root = '..\\model\\data\\'):
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


if __name__ == '__main__':
    reeval_sim()

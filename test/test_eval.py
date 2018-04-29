import torch as th
import sys
import os
sys.path.append('../')
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import evaluation


def test_evaluate():
    path = './model/noun_unknown/nouns.pth'
    ev = evaluation.Evaluator.initialize_by_file(path)
    print('exp', ev.evaluate(method='exp'))
    print('reci', ev.evaluate(method='reciprocal'))
    print('tanh', ev.evaluate(method='tanh'))
    print('neg', ev.evaluate(method='neg'))

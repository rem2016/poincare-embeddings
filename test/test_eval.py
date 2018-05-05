import torch as th
import re_eval
import sys
import os
sys.path.append('../')
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import evaluation


def test_evaluate():
    path = '../model/noun_unknown/nouns.pth'
    ev = evaluation.Evaluator.initialize_by_file(path)
    print('exp', ev.evaluate(method='exp'))
    print('reci', ev.evaluate(method='reciprocal'))
    print('tanh', ev.evaluate(method='tanh'))
    print('neg', ev.evaluate(method='neg'))


def test_all():
    models = re_eval.load_all_important_models('../model/data/', threshold=20, is_module_dir=False)
    print(len(models))
    for ky, model in models.items():
        if 'mammal' in ky:
            continue
        print('======================================================='*2)
        print('                ', ky)
        print('======================================================='*2)
        try:
            print(model.name)
            print(model.log.config)
            for epoch, m in model.load_k_models(5):
                ev = evaluation.Evaluator.initialize_by_file(m, model.log.k)
                print(epoch)
                # print('Sense', ev.evaluate(try_use_word=False, is_word_level=False))
                if not model.log.config.get('w2v_sim', False) and not model.log.config.get('word', False):
                    continue
                if not ev.word_index:
                    continue
                try:
                    print('K =', model.log.k)
                    print('Word', ev.evaluate(try_use_word=True, is_word_level=True), 'k=', ev.k)
                    print("Failed", ev.failed_times)
                    ev.k = 3
                    print('Word', ev.evaluate(try_use_word=True, is_word_level=True), 'k=', ev.k)
                    ev.k = 5
                    print('Word', ev.evaluate(try_use_word=True, is_word_level=True), 'k=', ev.k)
                except evaluation.OutofdateVersionError:
                    print("out of date")
                    break
        except ValueError:
            pass
        print('=======================================================')
        print()
        print()


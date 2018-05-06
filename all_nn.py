import torch.multiprocessing as mp
import numpy as np
from collections import defaultdict
import os
import sys
import sematch.evaluation as ev
import numpy as np
from time import time
import data
from numpy.linalg import norm
import logging
from word_vec_loader import WordVectorLoader
from collections import defaultdict


def save(path, ans, index2word):
    with open(path, 'w') as f:
        for a, items in ans.items():
            for b, sim in items.items():
                f.write(f'{index2word[a]}\t{index2word[b]}\t{sim}\n')


def control(queue, path, index2word, log):
    ans = {}
    target = 100
    step = 100
    start = time()
    while True:
        msg = queue.get()
        if msg is None:
            break
        if not isinstance(msg, dict):
            print(msg)
            break
        for key in msg.keys():
            ans[key] = msg[key]
        used = time() - start
        if len(ans) >= target:
            save(path, ans, index2word)
            target += step
        if len(ans) >= len(index2word):
            print("=================================")
            print("Finished!")
            print("=================================")
            break
        left_time = used / max(1, (len(ans))) * len(index2word) - used
        log.info(f"Left time {left_time}")


def calc(queue, start, end, _data, save_step, rank):
    save_dict = defaultdict(lambda: {})
    for i in range(start, end):
        for j in range(i, len(_data)):
            sim = calc_dist(_data[i], _data[j])
            if is_perfect_sim(sim):
                save_dict[i][j] = float(sim)
        if len(save_dict) % save_step == save_step - 1:
            queue.put(dict(save_dict))
            save_dict.clear()
    queue.put(dict(save_dict))


def calc_all(queue, start, end, _data, save_step, all_index):
    save_dict = defaultdict(lambda: {})
    for i in all_index[start: end]:
        for j in all_index:
            sim = calc_dist(_data[i], _data[j])
            save_dict[i][j] = float(sim)
        if len(save_dict) % save_step == save_step - 1:
            queue.put(dict(save_dict))
            save_dict.clear()
    queue.put(dict(save_dict))


def get_all_eval_words():
    dat = ev.WordSimDataset()
    data_word_noun = ['noun_rg', 'noun_mc', 'noun_ws353', 'noun_ws353-sim', 'noun_simlex']
    all_words = set()
    for dat_name in data_word_noun:
        words, sim = dat.load_dataset(dat_name)
        for a, b in words:
            all_words.add(a)
            all_words.add(b)
    return all_words


def is_good_sim(sim):
    # return sim < -0.2 or sim > 0.3  # total 5%
    return sim < -0.1 or sim > 0.2  # total 20%


def is_perfect_sim(sim):
    return sim > 0.5  # total 0.5%


def calc_dist(a, b):
    sim = np.sum(a * b) / (norm(a) * norm(b))
    return sim


def main(n_proc=1, save_path='all_nn.tsv', save_step=100):
    data_path = './wordnet/noun_closure.tsv'
    log = logging.getLogger('la')
    print(data_path)
    idx, objs, dwords = data.slurp(data_path, load_word=True, build_word_vector=True)
    print("loaded all words")

    vec = WordVectorLoader.word_vec
    step = int(len(vec) / n_proc + 0.999999999999999)
    processes = []
    queue = mp.Manager().Queue()
    for i in range(n_proc):
        p = mp.Process(target=calc, args=(queue, step * i, step * (i + 1), vec, save_step, i + 1))
        p.start()
        processes.append(p)

    all_words = get_all_eval_words()
    all_index = [dwords[x] - len(objs) for x in all_words if x in dwords]
    step = int(len(all_index) / n_proc + 0.999999999999999)
    for i in range(n_proc):
        p = mp.Process(target=calc_all, args=(queue, step * i, step * (i + 1), vec, save_step, all_index))
        p.start()
        processes.append(p)

    index2word = WordVectorLoader.index2word[len(objs):]
    ctrl = mp.Process(target=control, args=(queue, save_path, index2word, log))
    ctrl.start()
    ctrl.join()


if __name__ == '__main__':
    main()

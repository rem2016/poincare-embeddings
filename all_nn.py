import torch.multiprocessing as mp
import numpy as np
from time import time
import data
from numpy.linalg import norm
from word_vec_loader import WordVectorLoader
from collections import defaultdict


def save(path, ans, objs):
    with open(path, 'w') as f:
        for a, items in ans.items():
            for b, sim in items.items():
                f.write(f'{objs[a]}\t{objs[b]}\t{sim}\n')


def control(queue, path, objs):
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
            save(path, ans, objs)
            target += step
        if len(ans) >= len(objs):
            print("=================================")
            print("Finished!")
            print("=================================")
            break
        print("Left time", used / max(1, (len(ans))) * len(objs) - used)


def calc(queue, start, end, _data, save_step, rank):
    save_dict = defaultdict(lambda: {})
    for i in range(start, end):
        for j in range(i, len(_data)):
            sim = calc_dist(_data[i], _data[j])
            if is_perfect_sim(sim):
                save_dict[i][j] = float(sim)
        if i % save_step == save_step - 1:
            queue.put(dict(save_dict))
            save_dict.clear()
    queue.put(dict(save_dict))


def is_good_sim(sim):
    # return sim < -0.2 or sim > 0.3  # total 5%
    return sim < -0.1 or sim > 0.2  # total 20%


def is_perfect_sim(sim):
    return sim > 0.4  # total 0.5%


def calc_dist(a, b):
    sim = np.sum(a * b) / (norm(a) * norm(b))
    return sim


def main(n_proc=3, save_path='all_nn.tsv', save_step=100):
    data_path = './wordnet/noun_closure.tsv'
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
    queue.put({})
    ctrl = mp.Process(target=control, args=(queue, save_path, objs))
    ctrl.start()
    ctrl.join()


if __name__ == '__main__':
    main()

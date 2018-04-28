#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import count
from collections import defaultdict as ddict
import numpy as np
import torch as th
import os
import argparse
import random


def parse_seperator(line, length, sep='\t'):
    d = line.strip().split(sep)
    if len(d) == length:
        w = 1
    elif len(d) == length + 1:
        w = int(d[-1])
        d = d[:-1]
    else:
        raise RuntimeError(f'Malformed input ({line.strip()})')
    return tuple(d) + (w,)


def parse_tsv(line, length=2):
    return parse_seperator(line, length, '\t')


def parse_space(line, length=2):
    return parse_seperator(line, length, ' ')


def iter_line(fname, fparse, length=2, comment='#'):
    with open(fname, 'r') as fin:
        for line in fin:
            if line[0] == comment:
                continue
            tpl = fparse(line, length=length)
            if tpl is not None:
                yield tpl


def intmap_to_list(d):
    arr = [None for _ in range(len(d))]
    for v, i in d.items():
        arr[i] = v
    assert not any(x is None for x in arr)
    return arr


def slurp(fin, fparse=parse_tsv, symmetrize=False):
    ecount = count()
    enames = ddict(ecount.__next__)

    subs = []
    for i, j, w in iter_line(fin, fparse, length=2):
        if i == j:
            continue
        subs.append((enames[i], enames[j], w))
        if symmetrize:
            subs.append((enames[j], enames[i], w))
    idx = th.from_numpy(np.array(subs, dtype=np.int))

    # freeze defaultdicts after training data and convert to arrays
    objects = intmap_to_list(dict(enames))
    print(f'slurp: objects={len(objects)}, edges={len(idx)}')
    return idx, objects


# randomly hold out links
def _get_root_and_leaf(idx):
    as_head = set()
    as_tail = set()
    max_ = 0
    for row in idx:
        h, t, _ = row
        as_head.add(int(h))
        as_tail.add(int(t))
        max_ = max(max_, h, t)
        
    u = set(range(max_))
    root_and_leaf = (u - as_head) | (u - as_tail)
    return root_and_leaf


def split_data(fin, fout, max_test_rate=0.3):
    idx, objs = slurp(fin)
    root_and_leaf = _get_root_and_leaf(idx)
    train, test = [], []
    max_test_size = int(len(idx) * max_test_rate + 0.5)
    random.shuffle(idx)
    for h, t, _ in idx:
        t = int(t)
        h = int(h)
        if t not in root_and_leaf and h not in root_and_leaf and len(test) < max_test_size:
            test.append((objs[h], objs[t]))
        else:
            train.append((objs[h], objs[t]))

    with open(f'{fout}.train.tsv', 'w') as f:
        f.write('\n'.join(['\t'.join(x) for x in train]))
    with open(f'{fout}.test.tsv', 'w') as f:
        f.write('\n'.join(['\t'.join(x) for x in test]))
            
    return train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split Data')
    parser.add_argument('-input', help='Input data path', type=str, required=True)
    parser.add_argument('-output', help='Output data path', type=str, required=True)
    opt = parser.parse_args()
    split_data(opt.input, opt.output)
    
    
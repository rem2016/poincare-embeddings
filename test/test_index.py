import data
from nltk.corpus import wordnet as wn
from collections import Counter, defaultdict
import evaluation


from word_vec_loader import WordVectorLoader


def clear():
    WordVectorLoader.word2index = None
    WordVectorLoader.index2word = None
    WordVectorLoader.embeddings = None
    WordVectorLoader.sense_num = None


def assert_mapping_accord(dwords, wload):
    for i, v in enumerate(wload.index2word):
        if v is None:
            continue
        assert dwords[v] == i


def assert_mapping_back_correct(idx, objs, dwords, data_path):
    adj = defaultdict(lambda: set())
    for t, h, _ in idx:
        adj[h].add(t)

    links = defaultdict(lambda: set())
    with open(data_path, 'r') as f:
        for line in f:
            t, h = line[:-1].split('\t')
            links[t].add(h)

    for head_index, obj in enumerate(objs):
        this_syn = wn.synset(obj)
        for tail_index in adj[head_index]:
            if tail_index < len(objs):  # synset
                assert objs[head_index] in links[objs[tail_index]]
            else:  # word
                word = WordVectorLoader.index2word[tail_index]
                assert word in [x.name() for x in this_syn.lemmas()]


def test_slurp():
    clear()
    data_path = './wordnet/noun_closure.tsv'
    idx, objs, dwords = data.slurp(data_path, load_word=True, build_word_vector=True)
    total_num = len(objs) + len(dwords)
    count = Counter()

    # assert every node have links
    for t, h, _ in idx:
        count[t] += 1
        count[h] += 1
    for i in range(total_num):
        assert i in count

    assert_mapping_accord(dwords, WordVectorLoader)
    assert_mapping_back_correct(idx, objs, dwords, data_path)




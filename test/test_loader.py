import data_loader
from time import time
import data
from word_vec_loader import WordVectorLoader
from torch.utils.data import Dataset, DataLoader


def clear():
    WordVectorLoader.word2index = None
    WordVectorLoader.index2word = None
    WordVectorLoader.embeddings = None
    WordVectorLoader.sense_num = None


def test_load_debug():
    clear()
    data_path = '../wordnet/debug.tsv'
    idx, objs, dwords = data.slurp(data_path, load_word=True, build_word_vector=True)
    _data = data_loader.WordsDataset(WordVectorLoader.word_vec, len(objs))
    loader = DataLoader(
        _data,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=data_loader.SNGraphDataset.collate,
        timeout=200
    )
    for a, b in loader:
        print(a)
        print(b)

    print('a')


def test_load_mammals():
    clear()
    data_path = './wordnet/mammal_closure.tsv'
    start = time()
    idx, objs, dwords = data.slurp(data_path, load_word=True, build_word_vector=True)
    _data = data_loader.WordsDataset(WordVectorLoader.word_vec, len(objs))
    used = time() - start
    print("Loading used time", used)
    loader = DataLoader(
        _data,
        batch_size=20,
        shuffle=True,
        num_workers=2,
        collate_fn=data_loader.SNGraphDataset.collate,
        timeout=20
    )
    print("Average nn", _data.calc_word_average_adj())
    start = time()
    for a, b in loader:
        pass
    used = time() - start
    print(used)


def test_slurp_nouns():
    clear()
    data_path = '../wordnet/noun_closure.tsv'
    print(data_path)
    idx, objs, dwords = data.slurp(data_path, load_word=True, build_word_vector=True)


def test_load_nouns():
    clear()
    data_path = '../wordnet/noun_closure.tsv'
    print(data_path)
    idx, objs, dwords = data.slurp(data_path, load_word=True, build_word_vector=True)
    print("loaded all words")
    _data = data_loader.WordsDataset(WordVectorLoader.word_vec, len(objs))
    print("init _data")
    loader = DataLoader(
        _data,
        batch_size=200,
        shuffle=True,
        num_workers=0,
        collate_fn=data_loader.SNGraphDataset.collate
    )
    print("start looping...")
    start = time()
    last = time()
    for a, b in loader:
        print(time() - last)
        last = time()
    used = time() - start
    print(_data.calc_word_average_adj())
    print(used)

import sys
import torch as tf
sys.path.append('..')
from join_word2vec import WordVectorLoader


def test_word2vec():
    words = ['people', 'child', 'fish', 'swim', 'dance']
    words = {name: i for i, name in enumerate(words)}
    WordVectorLoader.build(words)
    v = WordVectorLoader.get_vec_by_word('child')
    assert isinstance(list(v), list)
    assert v.shape[0] == 300
    for word, index in WordVectorLoader.word2index.items():
        assert WordVectorLoader.index2word[index] == word
    assert len(WordVectorLoader.word2index) == len(words)
    assert WordVectorLoader.embeddings(tf.tensor([1, 2, 3])).shape == (3, 300)

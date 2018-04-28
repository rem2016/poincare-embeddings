import sys
sys.path.append('..')
import torch as th
import join_word2vec


def test_join():
    words = ['time', 'dog', 'weather', 'morning']
    join_word2vec.WordVectorLoader.build(words)
    model = join_word2vec.SNEmbeddingWithWord(3, 5, len(words), )
    print(model(th.tensor([0, 1, 2, 3, 4, 5])))



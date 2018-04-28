import spacy
from torch.nn.modules import Embedding
import torch as th


def from_pretrained(embeddings, freeze=True):
    r"""Creates Embedding instance from given 2-dimensional FloatTensor.

    Args:
        embeddings (Tensor): FloatTensor containing weights for the Embedding.
            First dimension is being passed to Embedding as 'num_embeddings', second as 'embedding_dim'.
        freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
            Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``

    Examples::

        >>> # FloatTensor containing pretrained weights
        >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
        >>> embedding = nn.Embedding.from_pretrained(weight)
        >>> # Get embeddings for index 1
        >>> input = torch.LongTensor([1])
        >>> embedding(input)
        tensor([[ 4.0000,  5.1000,  6.3000]])
    """
    assert embeddings.dim() == 2, \
        'Embeddings parameter is expected to be 2-dimensional'
    rows, cols = embeddings.shape
    embedding = Embedding(num_embeddings=rows, embedding_dim=cols, _weight=embeddings)
    embedding.weight.requires_grad = not freeze
    return embedding


class WordVectorLoader:
    word2index = None
    index2word = None
    word_vec = None
    embeddings = None
    sense_num = None

    def __init__(self):
        raise NotImplemented("Static class")

    @classmethod
    def build(cls, dwords):
        cls.sense_num = min(dwords.values())
        nlp = spacy.load('en_core_web_lg')
        word_vec = [None] * len(dwords)
        index2word = [None] * len(dwords)
        for word, i in dwords.items():
            i -= cls.sense_num
            word_vec[i] = nlp(word).vector
            index2word[i] = word

        assert all((x is not None for x in word_vec))
        cls.embeddings = from_pretrained(th.Tensor(word_vec), True)
        cls.index2word = index2word
        cls.word2index = dwords
        cls.word_vec = word_vec
        return cls.embeddings

    @classmethod
    def embed(cls, inputs):
        if cls.embeddings is None:
            raise ValueError()
        return cls.embeddings(inputs)

    @classmethod
    def get_vec_by_word(cls, w):
        return cls.word_vec[WordVectorLoader.word2index[w]]

    @classmethod
    def get_vec_by_index(cls, i):
        return cls.word_vec[i]


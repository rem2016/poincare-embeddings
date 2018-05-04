import data
from word_vec_loader import WordVectorLoader
from torch.utils.data import DataLoader
import data_loader


def test_you():
    data_path = '../wordnet/noun_closure.tsv'
    idx, objs, dwords = data.slurp(data_path, load_word=True, build_word_vector=True)
    la = data_loader.WordsDataset(WordVectorLoader.word_vec, WordVectorLoader.sense_num)
    datas = DataLoader(
        la,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        timeout=200
    )
    for index, v in datas:
        print(index)
        print(v)
        break


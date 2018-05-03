import embed
from word_vec_loader import WordVectorLoader
import train


def test_end2end():
    opt = embed.parse_opt(debug=True)
    log = embed.setup_log(opt, need_file=False)
    opt.dset = '../wordnet/debug.tsv'
    opt.dset_test = '../wordnet/debug.tsv'
    opt.ndproc = 0
    opt.debug = True
    opt.word = True
    opt.eval_each = 1
    args = embed.start_predicting(opt, log, debug=True)
    queue, log, train_adjacency, test_adjacency, data, _, distfn, opt.epochs, processes, _, _ = args
    msg = queue.get()
    epoch, elapsed, loss, model, word_sim_loss = msg
    assert model is not None
    weights = model.state_dict()['lr'].weights
    assert weights.gra




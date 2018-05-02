import evaluation
# This is so exciting
# The very moment before launching the rocket

def test_eval():
    d = r'C:\Users\Administrator\Documents\G\model\data\50d.sim\nouns.50d.train_w2vsim.lr=1.0.dim=50.negs=50.burnin=20.batch=50\519.nth'
    es = evaluation.Evaluator.initialize_by_file(d)
    print(es.rank())


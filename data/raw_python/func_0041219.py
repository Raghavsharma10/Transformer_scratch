def build_trainer(nn, ds, verbosity=1):
    """Configure neural net trainer from a pybrain dataset"""
    return pb.supervised.trainers.rprop.RPropMinusTrainer(nn, dataset=ds, batchlearning=True, verbose=bool(verbosity))
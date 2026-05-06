def sim_trainer(trainer, mean=0, std=1):
    """Simulate a trainer by activating its DataSet and returning DataFrame(columns=['Output','Target'])
    """
    return sim_network(network=trainer.module, ds=trainer.ds, mean=mean, std=std)
def trainer_results(trainer, mean=0, std=1, title='', show=True, save=True):
    """Plot the performance of the Network and SupervisedDataSet in a pybrain Trainer

    DataSet target and output values are denormalized before plotting with:

        output * std + mean

    Which inverses the normalization

        (output - mean) / std

    Args:
        trainer (Trainer): a pybrain Trainer instance containing a valid Network and DataSet
        ds (DataSet): a pybrain DataSet to override the one contained in `trainer`.
          Required if trainer is a Network instance rather than a Trainer instance.
        mean (float): mean of the denormalized dataset (default: 0)
          Only affects the scale of the plot
        std (float): std (standard deviation) of the denormalized dataset (default: 1)
        title (str): title to display on the plot.

    Returns:
        3-tuple: (trainer, mean, std), A trainer/dataset along with denormalization info
    """
    return plot_network_results(network=trainer.module, ds=trainer.ds, mean=mean, std=std, title=title,
                                show=show, save=save)
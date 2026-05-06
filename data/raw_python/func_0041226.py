def sim_network(network, ds=None, index=None, mean=0, std=1):
    """Simulate/activate a Network on a SupervisedDataSet and return DataFrame(columns=['Output','Target'])

    The DataSet's target and output values are denormalized before populating the dataframe columns:

        denormalized_output = normalized_output * std + mean

    Which inverses the normalization that produced the normalized output in the first place: \

        normalized_output = (denormalzied_output - mean) / std

    Args:
        network (Network): a pybrain Network instance to activate with the provided DataSet, `ds`
        ds (DataSet): a pybrain DataSet to activate the Network on to produce an output sequence
        mean (float): mean of the denormalized dataset (default: 0)
          Output is scaled
        std (float): std (standard deviation) of the denormalized dataset (default: 1)
        title (str): title to display on the plot.

    Returns:
        DataFrame: DataFrame with columns "Output" and "Target" suitable for df.plot-ting
    """
    # just in case network is a trainer or has a Module-derived instance as one of it's attribute
       # isinstance(network.module, (networks.Network, modules.Module))
    if hasattr(network, 'module') and hasattr(network.module, 'activate'):
        # may want to also check: isinstance(network.module, (networks.Network, modules.Module))
        network = network.module
    ds = ds or network.ds
    if not ds:
        raise RuntimeError("Unable to find a `pybrain.datasets.DataSet` instance to activate the Network with, "
                           " to plot the outputs. A dataset can be provided as part of a network instance or "
                           "as a separate kwarg if `network` is used to provide the `pybrain.Network`"
                           " instance directly.")
    results_generator = ((network.activate(ds['input'][i])[0] * std + mean, ds['target'][i][0] * std + mean)
                         for i in xrange(len(ds['input'])))

    return pd.DataFrame(results_generator, columns=['Output', 'Target'], index=index or range(len(ds['input'])))
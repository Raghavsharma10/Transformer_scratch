def _build_datasets(*args, **kwargs):
    """Build the datasets into a dict, where the keys are the name of the
    data set and the values are the data sets themselves.

    :param args:
        Tuple of unnamed data sets.
    :type args:
        `tuple` of varies
    :param kwargs:
        Dict of pre-named data sets.
    :type kwargs:
        `dict` of `unicode` to varies
    :return:
        The dataset dict.
    :rtype:
        `dict`
    """
    datasets = OrderedDict()
    _add_arg_datasets(datasets, args)
    _add_kwarg_datasets(datasets, kwargs)
    return datasets
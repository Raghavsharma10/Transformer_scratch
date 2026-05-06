def _add_arg_datasets(datasets, args):
    """Add data sets of the given args.

    :param datasets:
        The dict where to accumulate data sets.
    :type datasets:
        `dict`
    :param args:
        Tuple of unnamed data sets.
    :type args:
        `tuple` of varies
    """
    for dataset in args:
        # turn a value into a 1-tuple.
        if not isinstance(dataset, (tuple, GentyArgs)):
            dataset = (dataset,)

        # Create a test_name_suffix - basically the parameter list
        if isinstance(dataset, GentyArgs):
            dataset_strings = dataset     # GentyArgs supports iteration
        else:
            dataset_strings = [format_arg(data) for data in dataset]
        test_method_suffix = ", ".join(dataset_strings)

        datasets[test_method_suffix] = dataset
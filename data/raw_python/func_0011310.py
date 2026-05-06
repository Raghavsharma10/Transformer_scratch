def _add_kwarg_datasets(datasets, kwargs):
    """Add data sets of the given kwargs.

    :param datasets:
        The dict where to accumulate data sets.
    :type datasets:
        `dict`
    :param kwargs:
        Dict of pre-named data sets.
    :type kwargs:
        `dict` of `unicode` to varies
    """
    for test_method_suffix, dataset in six.iteritems(kwargs):
        datasets[test_method_suffix] = dataset
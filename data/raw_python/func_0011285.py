def _build_dataset_method(method, dataset):
    """
    Return a fabricated method that marshals the dataset into parameters
    for given 'method'
    :param method:
        The underlying test method.
    :type method:
        `callable`
    :param dataset:
        Tuple or GentyArgs instance containing the args of the dataset.
    :type dataset:
        `tuple` or :class:`GentyArgs`
    :return:
        Return an unbound function that will become a test method
    :rtype:
        `function`
    """
    if isinstance(dataset, GentyArgs):
        test_method = lambda my_self: method(
            my_self,
            *dataset.args,
            **dataset.kwargs
        )
    else:
        test_method = lambda my_self: method(
            my_self,
            *dataset
        )
    return test_method
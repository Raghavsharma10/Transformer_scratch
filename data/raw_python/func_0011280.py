def _expand_datasets(test_functions):
    """
    Generator producing test_methods, with an optional dataset.

    :param test_functions:
        Iterator over tuples of test name and test unbound function.
    :type test_functions:
        `iterator` of `tuple` of (`unicode`, `function`)
    :return:
        Generator yielding a tuple of
        - method_name      : Name of the test method
        - unbound function : Unbound function that will be the test method.
        - dataset name     : String representation of the given dataset
        - dataset          : Tuple representing the args for a test
        - param factory    : Function that returns params for the test method
    :rtype:
        `generator` of `tuple` of (
            `unicode`,
            `function`,
            `unicode` or None,
            `tuple` or None,
            `function` or None,
        )
    """
    for name, func in test_functions:

        dataset_tuples = chain(
            [(None, getattr(func, 'genty_datasets', {}))],
            getattr(func, 'genty_dataproviders', []),
        )

        no_datasets = True
        for dataprovider, datasets in dataset_tuples:
            for dataset_name, dataset in six.iteritems(datasets):
                no_datasets = False
                yield name, func, dataset_name, dataset, dataprovider

        if no_datasets:
            # yield the original test method, unaltered
            yield name, func, None, None, None
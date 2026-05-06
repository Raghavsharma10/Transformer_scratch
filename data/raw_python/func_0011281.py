def _expand_repeats(test_functions):
    """
    Generator producing test_methods, with any repeat count unrolled.

    :param test_functions:
        Sequence of tuples of
        - method_name      : Name of the test method
        - unbound function : Unbound function that will be the test method.
        - dataset name     : String representation of the given dataset
        - dataset          : Tuple representing the args for a test
        - param factory    : Function that returns params for the test method
    :type test_functions:
        `iterator` of `tuple` of
        (`unicode`, `function`, `unicode` or None, `tuple` or None, `function`)
    :return:
        Generator yielding a tuple of
        (method_name, unbound function, dataset, name dataset, repeat_suffix)
    :rtype:
        `generator` of `tuple` of (`unicode`, `function`,
        `unicode` or None, `tuple` or None, `function`, `unicode`)
    """
    for name, func, dataset_name, dataset, dataprovider in test_functions:
        repeat_count = getattr(func, 'genty_repeat_count', 0)
        if repeat_count:
            for i in range(1, repeat_count + 1):
                repeat_suffix = _build_repeat_suffix(i, repeat_count)
                yield (
                    name,
                    func,
                    dataset_name,
                    dataset,
                    dataprovider,
                    repeat_suffix,
                )
        else:
            yield name, func, dataset_name, dataset, dataprovider, None
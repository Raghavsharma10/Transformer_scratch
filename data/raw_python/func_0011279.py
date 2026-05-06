def genty(target_cls):
    """
    This decorator takes the information provided by @genty_dataset,
    @genty_dataprovider, and @genty_repeat and generates the corresponding
    test methods.

    :param target_cls:
        Test class whose test methods have been decorated.
    :type target_cls:
        `class`
    """
    tests = _expand_tests(target_cls)
    tests_with_datasets = _expand_datasets(tests)
    tests_with_datasets_and_repeats = _expand_repeats(tests_with_datasets)

    _add_new_test_methods(target_cls, tests_with_datasets_and_repeats)

    return target_cls
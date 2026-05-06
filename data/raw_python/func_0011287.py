def _add_method_to_class(
        target_cls,
        method_name,
        func,
        dataset_name,
        dataset,
        dataprovider,
        repeat_suffix,
):
    """
    Add the described method to the given class.

    :param target_cls:
        Test class to which to add a method.
    :type target_cls:
        `class`
    :param method_name:
        Base name of the method to add.
    :type method_name:
        `unicode`
    :param func:
        The underlying test function to call.
    :type func:
        `callable`
    :param dataset_name:
        Base name of the data set.
    :type dataset_name:
        `unicode` or None
    :param dataset:
        Tuple containing the args of the dataset.
    :type dataset:
        `tuple` or None
    :param repeat_suffix:
        Suffix to append to the name of the generated method.
    :type repeat_suffix:
        `unicode` or None
    :param dataprovider:
        The unbound function that's responsible for generating the actual
        params that will be passed to the test function. Can be None.
    :type dataprovider:
        `callable`
    """
    # pylint: disable=too-many-arguments
    test_method_name_for_dataset = _build_final_method_name(
        method_name,
        dataset_name,
        dataprovider.__name__ if dataprovider else None,
        repeat_suffix,
    )

    test_method_for_dataset = _build_test_method(func, dataset, dataprovider)

    test_method_for_dataset = functools.update_wrapper(
        test_method_for_dataset,
        func,
    )

    test_method_name_for_dataset = encode_non_ascii_string(
        test_method_name_for_dataset,
    )
    test_method_for_dataset.__name__ = test_method_name_for_dataset
    test_method_for_dataset.genty_generated_test = True

    # Add the method to the class under the proper name
    setattr(target_cls, test_method_name_for_dataset, test_method_for_dataset)
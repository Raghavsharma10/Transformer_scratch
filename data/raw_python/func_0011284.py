def _build_final_method_name(
        method_name,
        dataset_name,
        dataprovider_name,
        repeat_suffix,
):
    """
    Return a nice human friendly name, that almost looks like code.

    Example: a test called 'test_something' with a dataset of (5, 'hello')
         Return:  "test_something(5, 'hello')"

    Example: a test called 'test_other_stuff' with dataset of (9) and repeats
         Return: "test_other_stuff(9) iteration_<X>"

    :param method_name:
        Base name of the method to add.
    :type method_name:
        `unicode`
    :param dataset_name:
        Base name of the data set.
    :type dataset_name:
        `unicode` or None
    :param dataprovider_name:
        If there's a dataprovider involved, then this is its name.
    :type dataprovider_name:
        `unicode` or None
    :param repeat_suffix:
        Suffix to append to the name of the generated method.
    :type repeat_suffix:
        `unicode` or None
    :return:
        The fully composed name of the generated test method.
    :rtype:
        `unicode`
    """
    # For tests using a dataprovider, append "_<dataprovider_name>" to
    #  the test method name
    suffix = ''
    if dataprovider_name:
        suffix = '_{0}'.format(dataprovider_name)

    if not dataset_name and not repeat_suffix:
        return '{0}{1}'.format(method_name, suffix)

    if dataset_name:
        # Nosetest multi-processing code parses the full test name
        # to discern package/module names. Thus any periods in the test-name
        # causes that code to fail. So replace any periods with the unicode
        # middle-dot character. Yes, this change is applied independent
        # of the test runner being used... and that's fine since there is
        # no real contract as to how the fabricated tests are named.
        dataset_name = dataset_name.replace('.', REPLACE_FOR_PERIOD_CHAR)

    # Place data_set info inside parens, as if it were a function call
    suffix = '{0}({1})'.format(suffix, dataset_name or "")

    if repeat_suffix:
        suffix = '{0} {1}'.format(suffix, repeat_suffix)

    test_method_name_for_dataset = "{0}{1}".format(
        method_name,
        suffix,
    )

    return test_method_name_for_dataset
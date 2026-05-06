def contains(ref_value):
    """
    'Contains' validation_function generator.
    Returns a validation_function to check that `ref_value in x`

    :param ref_value: a value that must be present in x
    :return:
    """
    def contains_ref_value(x):
        if ref_value in x:
            return True
        else:
            raise DoesNotContainValue(wrong_value=x, ref_value=ref_value)

    contains_ref_value.__name__ = 'contains_{}'.format(ref_value)
    return contains_ref_value
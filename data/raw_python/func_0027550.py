def has_length(ref_length):
    """
    'length equals' validation function generator.
    Returns a validation_function to check that `len(x) == ref_length`

    :param ref_length:
    :return:
    """
    def has_length_(x):
        if len(x) == ref_length:
            return True
        else:
            raise WrongLength(wrong_value=x, ref_length=ref_length)

    has_length_.__name__ = 'length_equals_{}'.format(ref_length)
    return has_length_
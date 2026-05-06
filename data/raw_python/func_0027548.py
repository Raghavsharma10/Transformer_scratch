def minlen(min_length,
           strict=False  # type: bool
           ):
    """
    'Minimum length' validation_function generator.
    Returns a validation_function to check that len(x) >= min_length (strict=False, default)
    or len(x) > min_length (strict=True)

    :param min_length: minimum length for x
    :param strict: Boolean flag to switch between len(x) >= min_length (strict=False) and len(x) > min_length
    (strict=True)
    :return:
    """
    if strict:
        def minlen_(x):
            if len(x) > min_length:
                return True
            else:
                # raise Failure('minlen: len(x) > ' + str(min_length) + ' does not hold for x=' + str(x))
                raise TooShort(wrong_value=x, min_length=min_length, strict=True)
    else:
        def minlen_(x):
            if len(x) >= min_length:
                return True
            else:
                # raise Failure('minlen: len(x) >= ' + str(min_length) + ' does not hold for x=' + str(x))
                raise TooShort(wrong_value=x, min_length=min_length, strict=False)

    minlen_.__name__ = 'length_{}greater_than_{}'.format('strictly_' if strict else '', min_length)
    return minlen_
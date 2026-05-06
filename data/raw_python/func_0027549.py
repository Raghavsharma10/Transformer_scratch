def maxlen(max_length,
           strict=False  # type: bool
           ):
    """
    'Maximum length' validation_function generator.
    Returns a validation_function to check that len(x) <= max_length (strict=False, default) or len(x) < max_length (strict=True)

    :param max_length: maximum length for x
    :param strict: Boolean flag to switch between len(x) <= max_length (strict=False) and len(x) < max_length
    (strict=True)
    :return:
    """
    if strict:
        def maxlen_(x):
            if len(x) < max_length:
                return True
            else:
                # raise Failure('maxlen: len(x) < ' + str(max_length) + ' does not hold for x=' + str(x))
                raise TooLong(wrong_value=x, max_length=max_length, strict=True)
    else:
        def maxlen_(x):
            if len(x) <= max_length:
                return True
            else:
                # raise Failure('maxlen: len(x) <= ' + str(max_length) + ' does not hold for x=' + str(x))
                raise TooLong(wrong_value=x, max_length=max_length, strict=False)

    maxlen_.__name__ = 'length_{}lesser_than_{}'.format('strictly_' if strict else '', max_length)
    return maxlen_
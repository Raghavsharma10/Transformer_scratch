def length_between(min_len,
                   max_len,
                   open_left=False,  # type: bool
                   open_right=False  # type: bool
                   ):
    """
    'Is length between' validation_function generator.
    Returns a validation_function to check that `min_len <= len(x) <= max_len (default)`. `open_right` and `open_left`
    flags allow to transform each side into strict mode. For example setting `open_left=True` will enforce
    `min_len < len(x) <= max_len`.

    :param min_len: minimum length for x
    :param max_len: maximum length for x
    :param open_left: Boolean flag to turn the left inequality to strict mode
    :param open_right: Boolean flag to turn the right inequality to strict mode
    :return:
    """
    if open_left and open_right:
        def length_between_(x):
            if (min_len < len(x)) and (len(x) < max_len):
                return True
            else:
                # raise Failure('length between: {} < len(x) < {} does not hold for x={}'.format(min_len, max_len,
                # x))
                raise LengthNotInRange(wrong_value=x, min_length=min_len, left_strict=True, max_length=max_len,
                                       right_strict=True)
    elif open_left:
        def length_between_(x):
            if (min_len < len(x)) and (len(x) <= max_len):
                return True
            else:
                # raise Failure('length between: {} < len(x) <= {} does not hold for x={}'.format(min_len, max_len,
                # x))
                raise LengthNotInRange(wrong_value=x, min_length=min_len, left_strict=True, max_length=max_len,
                                       right_strict=False)
    elif open_right:
        def length_between_(x):
            if (min_len <= len(x)) and (len(x) < max_len):
                return True
            else:
                # raise Failure('length between: {} <= len(x) < {} does not hold for x={}'.format(min_len, max_len,
                #  x))
                raise LengthNotInRange(wrong_value=x, min_length=min_len, left_strict=False, max_length=max_len,
                                       right_strict=True)
    else:
        def length_between_(x):
            if (min_len <= len(x)) and (len(x) <= max_len):
                return True
            else:
                # raise Failure('length between: {} <= len(x) <= {} does not hold for x={}'.format(min_len,
                #  max_len, x))
                raise LengthNotInRange(wrong_value=x, min_length=min_len, left_strict=False, max_length=max_len,
                                       right_strict=False)

    length_between_.__name__ = 'length_between_{}_and_{}'.format(min_len, max_len)
    return length_between_
def between(min_val,          # type: Any
            max_val,          # type: Any
            open_left=False,  # type: bool
            open_right=False  # type: bool
            ):
    """
    'Is between' validation_function generator.
    Returns a validation_function to check that min_val <= x <= max_val (default). open_right and open_left flags allow
    to transform each side into strict mode. For example setting open_left=True will enforce min_val < x <= max_val

    :param min_val: minimum value for x
    :param max_val: maximum value for x
    :param open_left: Boolean flag to turn the left inequality to strict mode
    :param open_right: Boolean flag to turn the right inequality to strict mode
    :return:
    """
    if open_left and open_right:
        def between_(x):
            if (min_val < x) and (x < max_val):
                return True
            else:
                # raise Failure('{} < x < {} does not hold for x={}'.format(min_val, max_val, x))
                raise NotInRange(wrong_value=x, min_value=min_val, left_strict=True,
                                 max_value=max_val, right_strict=True)
    elif open_left:
        def between_(x):
            if (min_val < x) and (x <= max_val):
                return True
            else:
                # raise Failure('between: {} < x <= {} does not hold for x={}'.format(min_val, max_val, x))
                raise NotInRange(wrong_value=x, min_value=min_val, left_strict=True,
                                 max_value=max_val, right_strict=False)
    elif open_right:
        def between_(x):
            if (min_val <= x) and (x < max_val):
                return True
            else:
                # raise Failure('between: {} <= x < {} does not hold for x={}'.format(min_val, max_val, x))
                raise NotInRange(wrong_value=x, min_value=min_val, left_strict=False,
                                 max_value=max_val, right_strict=True)
    else:
        def between_(x):
            if (min_val <= x) and (x <= max_val):
                return True
            else:
                # raise Failure('between: {} <= x <= {} does not hold for x={}'.format(min_val, max_val, x))
                raise NotInRange(wrong_value=x, min_value=min_val, left_strict=False,
                                 max_value=max_val, right_strict=False)

    between_.__name__ = 'between_{}_and_{}'.format(min_val, max_val)
    return between_
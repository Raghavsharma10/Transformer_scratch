def gt(min_value,    # type: Any
       strict=False  # type: bool
       ):
    """
    'Greater than' validation_function generator.
    Returns a validation_function to check that x >= min_value (strict=False, default) or x > min_value (strict=True)

    :param min_value: minimum value for x
    :param strict: Boolean flag to switch between x >= min_value (strict=False) and x > min_value (strict=True)
    :return:
    """
    if strict:
        def gt_(x):
            if x > min_value:
                return True
            else:
                # raise Failure('x > ' + str(min_value) + ' does not hold for x=' + str(x))
                # '{val} is not strictly greater than {ref}'
                raise TooSmall(wrong_value=x, min_value=min_value, strict=True)
    else:
        def gt_(x):
            if x >= min_value:
                return True
            else:
                # raise Failure('x >= ' + str(min_value) + ' does not hold for x=' + str(x))
                # '{val} is not greater than {ref}'
                raise TooSmall(wrong_value=x, min_value=min_value, strict=False)

    gt_.__name__ = '{}greater_than_{}'.format('strictly_' if strict else '', min_value)
    return gt_
def lt(max_value,    # type: Any
       strict=False  # type: bool
       ):
    """
    'Lesser than' validation_function generator.
    Returns a validation_function to check that x <= max_value (strict=False, default) or x < max_value (strict=True)

    :param max_value: maximum value for x
    :param strict: Boolean flag to switch between x <= max_value (strict=False) and x < max_value (strict=True)
    :return:
    """
    if strict:
        def lt_(x):
            if x < max_value:
                return True
            else:
                # raise Failure('x < ' + str(max_value) + ' does not hold for x=' + str(x))
                # '{val} is not strictly lesser than {ref}'
                raise TooBig(wrong_value=x, max_value=max_value, strict=True)
    else:
        def lt_(x):
            if x <= max_value:
                return True
            else:
                # raise Failure('x <= ' + str(max_value) + ' does not hold for x=' + str(x))
                # '{val} is not lesser than {ref}'
                raise TooBig(wrong_value=x, max_value=max_value, strict=False)

    lt_.__name__ = '{}lesser_than_{}'.format('strictly_' if strict else '', max_value)
    return lt_
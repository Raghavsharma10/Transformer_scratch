def _round_and_pad(value, decimals):
    """
    Round the input value to `decimals` places, and pad with 0's
    if the resulting value is less than `decimals`.

    :param value:
        The value to round
    :param decimals:
        Number of decimals places which should be displayed after the rounding.
    :return:
        str of the rounded value
    """
    if isinstance(value, int) and decimals != 0:
        # if we get an int coordinate and we have a non-zero value for
        # `decimals`, we want to create a float to pad out.
        value = float(value)

    elif decimals == 0:
        # if get a `decimals` value of 0, we want to return an int.
        return repr(int(round(value, decimals)))

    rounded = repr(round(value, decimals))
    rounded += '0' * (decimals - len(rounded.split('.')[1]))
    return rounded
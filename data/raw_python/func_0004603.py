def sorted_index(values, x):
    """
    For list, values, returns the index location of element x. If x does not exist will raise an error.

    :param values: list
    :param x: item
    :return: integer index
    """
    i = bisect_left(values, x)
    j = bisect_right(values, x)
    return values[i:j].index(x) + i
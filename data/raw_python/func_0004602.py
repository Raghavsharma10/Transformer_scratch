def sorted_exists(values, x):
    """
    For list, values, returns the insert position for item x and whether the item already exists in the list. This
    allows one function call to return either the index to overwrite an existing value in the list, or the index to
    insert a new item in the list and keep the list in sorted order.

    :param values: list
    :param x: item
    :return: (exists, index) tuple
    """
    i = bisect_left(values, x)
    j = bisect_right(values, x)
    exists = x in values[i:j]
    return exists, i
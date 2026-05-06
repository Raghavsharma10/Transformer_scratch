def expandValues(inputs, count, name):
    """Returns the input list with the length of `count`. If the
    list is [1] and the count is 3. [1,1,1] is returned. The list
    must be the count length or 1. Normally called from `expandParameters()`
    where `name` is the symbolic name of the input.
    """
    if len(inputs) == count:
        expanded = inputs
    elif len(inputs) == 1:
        expanded = inputs * count
    else:
        raise ValueError('Incompatible number of values for ' + name)
    return expanded
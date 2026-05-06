def _full_keys(keys, ndim):
    """
    Given keys such as those passed to ``__getitem__`` for an
    array of ndim, return a fully expanded tuple of keys.

    In all instances, the result of this operation should follow:

        array[keys] == array[_full_keys(keys, array.ndim)]

    """
    if not isinstance(keys, tuple):
        keys = (keys,)

    # Make keys mutable, and take a copy.
    keys = list(keys)

    # Count the number of keys which actually slice a dimension.
    n_keys_non_newaxis = len([key for key in keys if key is not np.newaxis])

    # Numpy allows an extra dimension to be an Ellipsis, we remove it here
    # if Ellipsis is in keys, if this doesn't trigger we will raise an
    # IndexError.
    is_ellipsis = [key is Ellipsis for key in keys]
    if n_keys_non_newaxis - 1 >= ndim and any(is_ellipsis):
        # Remove the left-most Ellipsis, as numpy does.
        keys.pop(is_ellipsis.index(True))
        n_keys_non_newaxis -= 1

    if n_keys_non_newaxis > ndim:
        raise IndexError('Dimensions are over specified for indexing.')

    lh_keys = []
    # Keys, with the last key first.
    rh_keys = []

    take_from_left = True
    while keys:
        if take_from_left:
            next_key = keys.pop(0)
            keys_list = lh_keys
        else:
            next_key = keys.pop(-1)
            keys_list = rh_keys

        if next_key is Ellipsis:
            next_key = slice(None)
            take_from_left = not take_from_left
        keys_list.append(next_key)

    middle = [slice(None)] * (ndim - n_keys_non_newaxis)
    return tuple(lh_keys + middle + rh_keys[::-1])
def remove_index_from_handle(handle_with_index):
    '''
    Returns index and handle separately, in a tuple.

    :handle_with_index: The handle string with an index (e.g.
        500:prefix/suffix)
    :return: index and handle as a tuple.
    '''

    split = handle_with_index.split(':')
    if len(split) == 2:
        split[0] = int(split[0])
        return split
    elif len(split) == 1:
        return (None, handle_with_index)
    elif len(split) > 2:
        raise handleexceptions.HandleSyntaxError(
            msg='Too many colons',
            handle=handle_with_index,
            expected_syntax='index:prefix/suffix')
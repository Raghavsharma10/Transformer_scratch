def sorted_list_indexes(list_to_sort, key=None, reverse=False):
    """
    Sorts a list but returns the order of the index values of the list for the sort and not the values themselves.
    For example is the list provided is ['b', 'a', 'c'] then the result will be [2, 1, 3]

    :param list_to_sort: list to sort
    :param key: if not None then a function of one argument that is used to extract a comparison key from each
                list element
    :param reverse: if True then the list elements are sorted as if each comparison were reversed.
    :return: list of sorted index values
    """
    if key is not None:
        def key_func(i):
            return key(list_to_sort.__getitem__(i))
    else:
        key_func = list_to_sort.__getitem__
    return sorted(range(len(list_to_sort)), key=key_func, reverse=reverse)
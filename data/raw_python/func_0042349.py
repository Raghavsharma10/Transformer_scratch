def dict_find(d, which_key):
    """
    Finds key values in a nested dictionary. Returns a tuple of the dictionary in which
    the key was found along with the value
    """
    # If the starting point is a list, iterate recursively over all values
    if isinstance(d, (list, tuple)):
        for i in d:
            for result in dict_find(i, which_key):
                yield result

    # Else, iterate over all key values of the dictionary
    elif isinstance(d, dict):
        for k, v in d.items():
            if k == which_key:
                yield d, v
            for result in dict_find(v, which_key):
                yield result
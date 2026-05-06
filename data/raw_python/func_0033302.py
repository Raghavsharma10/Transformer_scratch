def _add_to_dict(t, container, name, value):
    """
    Adds an item to a dictionary, or raises an exception if an item with the
    specified key already exists in the dictionary.
    """

    if name in container:
        raise Exception("%s '%s' already exists" % (t, name))
    else:
        container[name] = value
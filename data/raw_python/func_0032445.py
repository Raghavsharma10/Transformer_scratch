def deepcopy(value):
    """
    The default copy.deepcopy seems to copy all objects and some are not
    `copy-able`.

    We only need to make sure the provided data is a copy per key, object does
    not need to be copied.
    """

    if not isinstance(value, (dict, list, tuple)):
        return value

    if isinstance(value, dict):
        copy = {}
        for k, v in value.items():
            copy[k] = deepcopy(v)

    if isinstance(value, tuple):
        copy = list(range(len(value)))

        for k in get_keys(list(value)):
            copy[k] = deepcopy(value[k])

        copy = tuple(copy)

    if isinstance(value, list):
        copy = list(range(len(value)))

        for k in get_keys(value):
            copy[k] = deepcopy(value[k])

    return copy
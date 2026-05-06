def flatten(data, parent_key='', sep='_'):
    """
    Transform dictionary multilevel values to one level dict, concatenating
    the keys with sep between them.
    """
    items = []

    if isinstance(data, list):
        logger.debug('Flattening list {}'.format(data))
        list_keys = [str(i) for i in range(0, len(data))]
        items.extend(
            flatten(dict(zip(list_keys, data)), parent_key, sep=sep).items())

    elif isinstance(data, dict):
        logger.debug('Flattening dict {}'.format(data))

        for key, value in data.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, collections.MutableMapping):
                items.extend(flatten(value, new_key, sep=sep).items())
            else:
                if isinstance(value, list):
                    list_keys = [str(i) for i in range(0, len(value))]
                    items.extend(
                        flatten(
                            dict(zip(list_keys, value)), new_key, sep=sep).items())
                else:
                    items.append((new_key, value))
    else:
        logger.debug('Nothing to flatten with {}'.format(data))
        return data

    return collections.OrderedDict(items)
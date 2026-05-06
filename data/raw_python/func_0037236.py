def mget(m_dict, keys, default=None, delimiter=':'):
    """
    :param m_dict: A dictionary to search inside of
    :type m_dict: dict
    :param keys: A list of keys
    :type keys: str
    :param default: A default value to return if none found
    :param delimiter: The delimiter used in the keys list
    :type delimiter: str
    :return: The value according to the keys list
    """
    val = m_dict
    keys = keys.split(delimiter)
    for key in keys:
        try:
            val = val[key]
        except KeyError:
            return default
        except TypeError:
            return default
    return val
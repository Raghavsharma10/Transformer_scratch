def mset(m_dict, keys, value, delimiter=':'):
    """
    :param m_dict: A dictionary to set the value inside of
    :type m_dict: dict
    :param keys: A list of keys
    :type keys: str
    :param value: The value to set inside of the dictionary
    :param delimiter: The delimiter used in the keys list
    :type delimiter: str
    """
    val = m_dict
    keys = keys.split(delimiter)
    for i, key in enumerate(keys):
        try:
            if i == len(keys) - 1:
                val[key] = value
                return
            else:
                val = val[key]
        except KeyError:
            if i == len(keys) - 1:
                val[key] = value
                return
            else:
                val[key] = {}
                val = val[key]
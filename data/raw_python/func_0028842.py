def has(prop, object_or_dct):
    """
    Implementation of ramda has
    :param prop:
    :param object_or_dct:
    :return:
    """
    return prop in object_or_dct if isinstance(dict, object_or_dct) else hasattr(object_or_dct, prop)
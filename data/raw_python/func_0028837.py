def prop_eq_or(default, key, value, dct):
    """
        Ramda propEq plus propOr implementation
    :param default:
    :param key:
    :param value:
    :param dct:
    :return:
    """
    return dct[key] and dct[key] == value if key in dct else default
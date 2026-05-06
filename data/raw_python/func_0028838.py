def prop_eq_or_in_or(default, key, value, dct):
    """
        Ramda propEq/propIn plus propOr
    :param default:
    :param key:
    :param value:
    :param dct:
    :return:
    """
    return has(key, dct) and \
           (dct[key] == value if key in dct else (
               dct[key] in value if isinstance((list, tuple), value) and not isinstance(str, value)
               else default
           ))
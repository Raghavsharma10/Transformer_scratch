def decorator_info(f_name, expected, actual, flag):
    """
    Convenience function returns nicely formatted error/warning msg.
    :param f_name:
    :param expected:
    :param actual:
    :param flag:
    :return:
    """
    format = lambda types: ', '.join([str(t).split("'")[1] for t in types])
    expected, actual = format(expected), format(actual)
    msg = "'{}' method ".format(f_name) \
          + ("accepts", "returns")[flag] + " ({}), but ".format(expected) \
          + ("was given", "result is")[flag] + " ({})".format(actual)

    return msg
def _is_referenced_in_argv(method_name):
    """
    Various test runners allow one to run a specific test like so:
        python -m unittest -v <test_module>.<test_name>

    Return True is the given method name is so referenced.

    :param method_name:
        Base name of the method to add.
    :type method_name:
        `unicode`
    :return:
        Is the given method referenced by the command line.
    :rtype:
        `bool`
    """
    expr = '.*[:.]{0}$'.format(method_name)
    regex = re.compile(expr)
    return any(regex.match(arg) for arg in sys.argv)
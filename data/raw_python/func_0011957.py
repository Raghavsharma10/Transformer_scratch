def makename(package, module):
    """Join package and module with a dot.

    Package or Module can be empty.

    :param package: the package name
    :type package: :class:`str`
    :param module: the module name
    :type module: :class:`str`
    :returns: the joined name
    :rtype: :class:`str`
    :raises: :class:`AssertionError`, if both package and module are empty
    """
    # Both package and module can be None/empty.
    assert package or module, "Specify either package or module"
    if package:
        name = package
        if module:
            name += '.' + module
    else:
        name = module
    return name
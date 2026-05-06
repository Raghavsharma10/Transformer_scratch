def get_subpackages(app, module):
    """Get all subpackages for the given module/package

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param module: the module to query or module path
    :type module: module | str
    :returns: list of packages names
    :rtype: list
    :raises: TypeError
    """
    submodules = _get_submodules(app, module)
    return [name for name, ispkg in submodules if ispkg]
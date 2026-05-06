def _get_submodules(app, module):
    """Get all submodules for the given module/package

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param module: the module to query or module path
    :type module: module | str
    :returns: list of module names and boolean whether its a package
    :rtype: list
    :raises: TypeError
    """
    if inspect.ismodule(module):
        if hasattr(module, '__path__'):
            p = module.__path__
        else:
            return []
    elif isinstance(module, str):
        p = module
    else:
        raise TypeError("Only Module or String accepted. %s given." % type(module))
    logger.debug('Getting submodules of %s', p)
    submodules = [(name, ispkg) for loader, name, ispkg in pkgutil.iter_modules(p)]
    logger.debug('Found submodules of %s: %s', module, submodules)
    return submodules
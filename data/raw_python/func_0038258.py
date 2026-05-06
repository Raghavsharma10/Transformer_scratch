def _recursive_import(package):
    """
    Args:
        package(py:term:`package`): Package to walk

    Import all modules from a package recursively
    """

    prefix = '%s.' % (package.__name__)

    path = getattr(package, '__path__', None)

    if path:
        for submod in pkgutil.walk_packages(path, prefix=prefix):
            _import_module(submod[1], submod[0].path)
def check_python_import(package_or_module):
    '''
    Checks if a python package or module is importable.
    Arguments:
        package_or_module -- the package or module name to check
    Returns:
        True or False
    '''
    logger = logging.getLogger(__name__)
    logger.debug("Checking python import '%s'...", package_or_module)
    loader = pkgutil.get_loader(package_or_module)
    found = loader is not None
    if found:
        logger.debug("Python %s '%s' found",
                     "package" if loader.is_package(package_or_module)
                     else "module", package_or_module)
    else:  # pragma: no cover
        logger.debug("Python import '%s' not found", package_or_module)
    return found
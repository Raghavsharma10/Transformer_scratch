def import_name(app, name):
    """Import the given name and return name, obj, parent, mod_name

    :param name: name to import
    :type name: str
    :returns: the imported object or None
    :rtype: object | None
    :raises: None
    """
    try:
        logger.debug('Importing %r', name)
        name, obj = autosummary.import_by_name(name)[:2]
        logger.debug('Imported %s', obj)
        return obj
    except ImportError as e:
        logger.warn("Jinjapidoc failed to import %r: %s", name, e)
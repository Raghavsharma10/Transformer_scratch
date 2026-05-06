def shall_skip(app, module, private):
    """Check if we want to skip this module.

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param module: the module name
    :type module: :class:`str`
    :param private: True, if privates are allowed
    :type private: :class:`bool`
    """
    logger.debug('Testing if %s should be skipped.', module)
    # skip if it has a "private" name and this is selected
    if module != '__init__.py' and module.startswith('_') and \
       not private:
        logger.debug('Skip %s because its either private or __init__.', module)
        return True
    logger.debug('Do not skip %s', module)
    return False
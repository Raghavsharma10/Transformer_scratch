def getLogger(name):
    """Return a logger from a given name.

    If the name does not have a log handler, this will create one for it based
    on the module name which will log everything to a log file in a location
    the executing user will have access to.

    :param name: ``str``
    :return: ``object``
    """
    log = logging.getLogger(name=name)
    for handler in log.handlers:
        if name == handler.name:
            return log
    else:
        return LogSetup().default_logger(name=name.split('.')[0])
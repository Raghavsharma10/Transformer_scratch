def loglevel(leveltype=None, isequal=False):
    """
    Set or get the logging level of Quilt

    :type leveltype: string or integer
    :param leveltype: Choose the logging level. Possible choices are none (0), debug (10), info (20), warning (30), error (40) and critical (50).

    :type isequal: boolean
    :param isequal: Check if level is equal to leveltype.

    :return: If the level is equal to leveltype.
    :rtype: boolean

    >>> loglevel()
    30
    """
    log = logging.getLogger(__name__)
    leveltype = leveltype
    loglevels = {
        "none": 0,
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40,
        "critical": 50
    }
    if leveltype is None and isequal is False:
        return log.getEffectiveLevel()
    if leveltype is not None and isequal is True:
        if leveltype in loglevels.values():
            return leveltype == log.getEffectiveLevel()
        elif leveltype in loglevels:
            return loglevels[leveltype] == log.getEffectiveLevel()
        raise ValueError(
            "Incorrect input provided. It should be none, debug, info, warning, error or critical."
        )
    if leveltype in loglevels.values():
        log.basicConfig(level=leveltype)
    elif leveltype in loglevels:
        log.basicConfig(level=loglevels[leveltype])
    else:
        raise ValueError(
            "Incorrect input provided. It should be none, debug, info, warning, error or critical."
        )
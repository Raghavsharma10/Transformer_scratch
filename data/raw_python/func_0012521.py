def get_registered_loggers(hide_children=False, hide_reusables=False):
    """
    Find the names of all loggers currently registered

    :param hide_children: only return top level logger names
    :param hide_reusables: hide the reusables loggers
    :return: list of logger names
    """

    return [logger for logger in logging.Logger.manager.loggerDict.keys()
            if not (hide_reusables and "reusables" in logger)
            and not (hide_children and "." in logger)]
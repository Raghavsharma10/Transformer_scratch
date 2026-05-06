def get_apps(exclude=(), append=(), current={'apps': INSTALLED_APPS}):
    """
    Returns INSTALLED_APPS without the apps listed in exclude and with the apps
    listed in append.

    The use of a mutable dict is intentional, in order to preserve the state of
    the INSTALLED_APPS tuple across multiple settings files.
    """

    current['apps'] = tuple(
        [a for a in current['apps'] if a not in exclude]
    ) + tuple(append)
    return current['apps']
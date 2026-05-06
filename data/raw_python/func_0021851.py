def install_backport_hook(api):
    """
    Install a backport import hook for Qt4 api

    Parameters
    ----------
    api : str
        The Qt4 api whose structure should be intercepted
        ('pyqt4' or 'pyside').

    Example
    -------
    >>> install_backport_hook("pyqt4")
    >>> import PyQt4
    Loaded module AnyQt._backport as a substitute for PyQt4

    """
    if api == USED_API:
        raise ValueError

    sys.meta_path.insert(0, ImportHookBackport(api))
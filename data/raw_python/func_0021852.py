def install_deny_hook(api):
    """
    Install a deny import hook for Qt api.

    Parameters
    ----------
    api : str
        The Qt api whose import should be prevented

    Example
    -------
    >>> install_deny_import("pyqt4")
    >>> import PyQt4
    Traceback (most recent call last):...
    ImportError: Import of PyQt4 is denied.

    """
    if api == USED_API:
        raise ValueError

    sys.meta_path.insert(0, ImportHookDeny(api))
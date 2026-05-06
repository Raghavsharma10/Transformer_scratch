def setpreferredapi(api):
    """
    Set the preferred Qt API.

    Will raise a RuntimeError if a Qt API was already selected.

    Note that QT_API environment variable (if set) will take precedence.
    """
    global __PREFERRED_API
    if __SELECTED_API is not None:
        raise RuntimeError("A Qt api {} was already selected"
                           .format(__SELECTED_API))

    if api.lower() not in {"pyqt4", "pyqt5", "pyside", "pyside2"}:
        raise ValueError(api)
    __PREFERRED_API = api.lower()
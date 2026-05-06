def selectapi(api):
    """
    Select an Qt API to use.

    This can only be set once and before any of the Qt modules are explicitly
    imported.
    """
    global __SELECTED_API, USED_API
    if api.lower() not in {"pyqt4", "pyqt5", "pyside", "pyside2"}:
        raise ValueError(api)

    if __SELECTED_API is not None and __SELECTED_API.lower() != api.lower():
        raise RuntimeError("A Qt API {} was already selected"
                           .format(__SELECTED_API))
    elif __SELECTED_API is None:
        __SELECTED_API = api.lower()
        from . import _api
        USED_API = _api.USED_API
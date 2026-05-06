def load_ctypes_library(name):
    """Takes a library name and calls find_library in case loading fails,
    since some girs don't include the real .so name.

    Raises OSError like LoadLibrary if loading fails.

    e.g. javascriptcoregtk-3.0 should be libjavascriptcoregtk-3.0.so on unix
    """

    try:
        return cdll.LoadLibrary(name)
    except OSError:
        name = find_library(name)
        if name is None:
            raise
        return cdll.LoadLibrary(name)
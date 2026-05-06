def get_winfunc(libname, funcname, restype=None, argtypes=(), _libcache={}):
    """Retrieve a function from a library/DLL, and set the data types."""
    if libname not in _libcache:
        _libcache[libname] = windll.LoadLibrary(libname)
    func = getattr(_libcache[libname], funcname)
    func.argtypes = argtypes
    func.restype = restype
    return func
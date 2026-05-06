def cxx(source, libraries=[]):
    r"""
    >>> cxx('extern "C" { int add(int a, int b) {return a + b;} }').add(40, 2)
    42
    """
    path = _cc_build_shared_lib(source, '.cc', libraries)
    return ctypes.cdll.LoadLibrary(path)
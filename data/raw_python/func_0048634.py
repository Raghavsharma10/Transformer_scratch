def c(source, libraries=[]):
    r"""
    >>> c('int add(int a, int b) {return a + b;}').add(40, 2)
    42
    >>> sqrt = c('''
    ... #include <math.h>
    ... double _sqrt(double x) {return sqrt(x);}
    ... ''', ['m'])._sqrt
    >>> sqrt.restype = ctypes.c_double
    >>> sqrt(ctypes.c_double(400.0))
    20.0
    """
    path = _cc_build_shared_lib(source, '.c', libraries)
    return ctypes.cdll.LoadLibrary(path)
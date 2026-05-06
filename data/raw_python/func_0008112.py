def set_implementation(impl):
    """
    Sets the implementation of this module

    Parameters
    ----------
    impl : str
        One of ["python", "c"]

    """
    global __impl__
    if impl.lower() == 'python':
        __impl__ = __IMPL_PYTHON__
    elif impl.lower() == 'c':
        __impl__ = __IMPL_C__
    else:
        import warnings
        warnings.warn('Implementation '+impl+' is not known. Using the fallback python implementation.')
        __impl__ = __IMPL_PYTHON__
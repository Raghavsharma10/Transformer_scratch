def _jit(function):
    """
    Compile a function using a jit compiler.

    The function is always compiled to check errors, but is only used outside
    tests, so that code coverage analysis can be performed in jitted functions.

    The tests set sys._called_from_test in conftest.py.

    """
    import sys

    compiled = numba.jit(function)

    if hasattr(sys, '_called_from_test'):
        return function
    else:  # pragma: no cover
        return compiled
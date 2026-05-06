def import_best(c_module, py_module):
    """ Import the best available module,
    with C preferred to pure Python.
    """
    from importlib import import_module
    from os import getenv
    pure_python = getenv("PURE_PYTHON", "")
    if pure_python:
        return import_module(py_module)
    else:
        try:
            return import_module(c_module)
        except ImportError:
            return import_module(py_module)
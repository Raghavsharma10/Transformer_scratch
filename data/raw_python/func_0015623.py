def get_backend(name):
    """Returns the backend by name or raises KeyError"""

    for backend in _BACKENDS:
        if backend.NAME == name:
            return backend
    raise KeyError("Backend %r not available" % name)
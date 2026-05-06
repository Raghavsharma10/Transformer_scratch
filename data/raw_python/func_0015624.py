def set_backend(name=None):
    """Set a prefered ffi backend (cffi, ctypes).

    set_backend() -- default
    set_backend("cffi") -- cffi first, others as fallback
    set_backend("ctypes") -- ctypes first, others as fallback
    """

    possible = list(_BACKENDS)
    if name is None:
        names = []
    else:
        names = name.split(",")

    for name in reversed(names):
        for backend in list(possible):
            if backend.NAME == name:
                possible.remove(backend)
                possible.insert(0, backend)
                break
        else:
            raise LookupError("Unkown backend: %r" % name)

    # only add null as fallback it explicitly specified
    if "null" not in names:
        possible = [b for b in possible if b.NAME != "null"]

    _ACTIVE_BACKENDS[:] = possible
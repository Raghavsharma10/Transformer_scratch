def method_header(method_name, nogil=False, idx_as_arg=False):
    """Returns the Cython method header for methods without arguments except
    `self`."""
    if not config.FASTCYTHON:
        nogil = False
    header = 'cpdef inline void %s(self' % method_name
    header += ', int idx)' if idx_as_arg else ')'
    header += ' nogil:' if nogil else ':'
    return header
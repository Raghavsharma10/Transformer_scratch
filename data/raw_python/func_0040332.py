def wraps(__fn, **kw):
        """Like ``functools.wraps``, with support for annotations."""
        kw['assigned'] = kw.get('assigned', WRAPPER_ASSIGNMENTS)
        return functools.wraps(__fn, **kw)
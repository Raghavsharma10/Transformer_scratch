def set_trace_cond(*args, **kw):
    """ Sets a condition for set_trace statements that have the
        specified marker.  A condition can either callable, in
        which case it should take one argument, which is the
        number of times set_trace(marker) has been called,
        or it can be a number, in which case the break will
        only be called.
    """
    for key, val in kw.items():
        Epdb.set_trace_cond(key, val)
    for arg in args:
        Epdb.set_trace_cond(arg, True)
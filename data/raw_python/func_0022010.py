def _daemon_thread(*a, **kw):
    """
    Create a `threading.Thread`, but always set ``daemon``.
    """
    thread = Thread(*a, **kw)
    thread.daemon = True
    return thread
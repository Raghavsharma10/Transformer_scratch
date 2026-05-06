def get_handler():
    """Return the handler as a named tuple.

    The named tuple attributes are 'host', 'port', 'signum'.
    Return None when no handler has been registered.
    """
    host, port, signum = _pdbhandler._registered()
    if signum:
        return Handler(host if host else DFLT_ADDRESS[0].encode(),
                       port if port else DFLT_ADDRESS[1], signum)
def proxy(host='localhost', port=4304, flags=0, persistent=False,
          verbose=False, ):
    """factory function that returns a proxy object for an owserver at
    host, port.
    """

    # resolve host name/port
    try:
        gai = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM,
                                 socket.IPPROTO_TCP)
    except socket.gaierror as err:
        raise ConnError(*err.args)

    # gai is a (non empty) list of tuples, search for the first working one
    assert gai
    for (family, _type, _proto, _, sockaddr) in gai:
        assert _type is socket.SOCK_STREAM and _proto is socket.IPPROTO_TCP
        owp = _Proxy(family, sockaddr, flags, verbose)
        try:
            # check if there is an owserver listening
            owp.ping()
        except ConnError as err:
            # no connection, go over to next sockaddr
            lasterr = err.args
            continue
        else:
            # ok, live owserver found, stop searching
            break
    else:
        # no server listening on (family, sockaddr) found:
        raise ConnError(*lasterr)

    # init errno to errmessage mapping
    # FIXME: should this be only optional?
    owp._init_errcodes()

    if persistent:
        owp = clone(owp, persistent=True)

    # here we should have all connections closed
    assert not isinstance(owp, _PersistentProxy) or owp.conn is None

    return owp
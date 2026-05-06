def connect(address, **config):
    """ Connect and perform a handshake and return a valid Connection object, assuming
    a protocol version can be agreed.
    """
    ssl_context = make_ssl_context(**config)
    last_error = None
    # Establish a connection to the host and port specified
    # Catches refused connections see:
    # https://docs.python.org/2/library/errno.html
    log_debug("[#0000]  C: <RESOLVE> %s", address)
    resolver = Resolver(custom_resolver=config.get("resolver"))
    resolver.addresses.append(address)
    resolver.custom_resolve()
    resolver.dns_resolve()
    for resolved_address in resolver.addresses:
        try:
            s = _connect(resolved_address, **config)
            s, der_encoded_server_certificate = _secure(s, address[0], ssl_context)
            connection = _handshake(s, resolved_address, der_encoded_server_certificate, **config)
        except Exception as error:
            last_error = error
        else:
            return connection
    if last_error is None:
        raise ServiceUnavailable("Failed to resolve addresses for %s" % address)
    else:
        raise last_error
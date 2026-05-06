def postman(host, port=587, auth=(None, None),
            force_tls=False, options=None):
    """
    Creates a Postman object with TLS and Auth
    middleware. TLS is placed before authentication
    because usually authentication happens and is
    accepted only after TLS is enabled.

    :param auth: Tuple of (username, password) to
        be used to ``login`` to the server.
    :param force_tls: Whether TLS should be forced.
    :param options: Dictionary of keyword arguments
        to be used when the SMTP class is called.
    """
    return Postman(
        host=host,
        port=port,
        middlewares=[
            middleware.tls(force=force_tls),
            middleware.auth(*auth),
        ],
        **options
    )
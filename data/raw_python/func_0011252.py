def ssl_wrap_socket(sock, keyfile=None, certfile=None, cert_reqs=None,
                    ca_certs=None, server_hostname=None,
                    ssl_version=None, ciphers=None, ssl_context=None):
    """
    All arguments except for server_hostname and ssl_context have the same
    meaning as they do when using :func:`ssl.wrap_socket`.

    :param server_hostname:
        When SNI is supported, the expected hostname of the certificate
    :param ssl_context:
        A pre-made :class:`SSLContext` object. If none is provided, one will
        be created using :func:`create_urllib3_context`.
    :param ciphers:
        A string of ciphers we wish the client to support. This is not
        supported on Python 2.6 as the ssl module does not support it.
    """
    context = ssl_context
    if context is None:
        context = create_urllib3_context(ssl_version, cert_reqs,
                                         ciphers=ciphers)

    if ca_certs:
        try:
            context.load_verify_locations(ca_certs)
        except IOError as e:  # Platform-specific: Python 2.6, 2.7, 3.2
            raise SSLError(e)
        # Py33 raises FileNotFoundError which subclasses OSError
        # These are not equivalent unless we check the errno attribute
        except OSError as e:  # Platform-specific: Python 3.3 and beyond
            if e.errno == errno.ENOENT:
                raise SSLError(e)
            raise
    if certfile:
        context.load_cert_chain(certfile, keyfile)
    if HAS_SNI:  # Platform-specific: OpenSSL with enabled SNI
        return context.wrap_socket(sock, server_hostname=server_hostname)
    return context.wrap_socket(sock)
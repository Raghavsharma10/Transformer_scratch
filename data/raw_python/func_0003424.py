def _get_server_certificate(addr, ssl_version=PROTOCOL_SSLv23, ca_certs=None):
    """Retrieve a server certificate

    Retrieve the certificate from the server at the specified address,
    and return it as a PEM-encoded string.
    If 'ca_certs' is specified, validate the server cert against it.
    If 'ssl_version' is specified, use it in the connection attempt.
    """

    if ssl_version not in (PROTOCOL_DTLS, PROTOCOL_DTLSv1, PROTOCOL_DTLSv1_2):
        return _orig_get_server_certificate(addr, ssl_version, ca_certs)

    if ca_certs is not None:
        cert_reqs = ssl.CERT_REQUIRED
    else:
        cert_reqs = ssl.CERT_NONE
    af = getaddrinfo(addr[0], addr[1])[0][0]
    s = ssl.wrap_socket(socket(af, SOCK_DGRAM),
                        ssl_version=ssl_version,
                        cert_reqs=cert_reqs, ca_certs=ca_certs)
    s.connect(addr)
    dercert = s.getpeercert(True)
    s.close()
    return ssl.DER_cert_to_PEM_cert(dercert)
def _get_certificate(cert_url):
    """Download and validate a specified Amazon PEM file."""
    global _cache

    if cert_url in _cache:
        cert = _cache[cert_url]
        if cert.has_expired():
            _cache = {}
        else:
            return cert

    url = urlparse(cert_url)
    host = url.netloc.lower()
    path = posixpath.normpath(url.path)

    # Sanity check location so we don't get some random person's cert.
    if url.scheme != 'https' or \
       host not in ['s3.amazonaws.com', 's3.amazonaws.com:443'] or \
       not path.startswith('/echo.api/'):
        log.error('invalid cert location %s', cert_url)
        return

    resp = urlopen(cert_url)
    if resp.getcode() != 200:
        log.error('failed to download certificate')
        return

    cert = crypto.load_certificate(crypto.FILETYPE_PEM, resp.read())

    if cert.has_expired() or cert.get_subject().CN != 'echo-api.amazon.com':
        log.error('certificate expired or invalid')
        return

    _cache[cert_url] = cert
    return cert
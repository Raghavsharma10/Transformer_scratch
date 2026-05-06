def _certifi_where_for_ssl_version():
    """Gets the right location for certifi certifications for the current SSL
    version.

    Older versions of SSL don't support the stronger set of root certificates.
    """
    if not ssl:
        return

    if ssl.OPENSSL_VERSION_INFO < (1, 0, 2):
        warnings.warn(
            'You are using an outdated version of OpenSSL that '
            'can\'t use stronger root certificates.')
        return certifi.old_where()

    return certifi.where()
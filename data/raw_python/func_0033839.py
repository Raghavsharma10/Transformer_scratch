def find_credential():
    """Locate the users X509 certificate and key files

    This method uses the C{X509_USER_CERT} and C{X509_USER_KEY} to locate
    valid proxy information. If those are not found, the standard location
    in /tmp/ is searched.

    @raises RuntimeError: if the proxy found via either method cannot
                          be validated
    @raises RuntimeError: if the cert and key files cannot be located
    """

    rfc_proxy_msg = ("Could not find a RFC 3820 compliant proxy credential."
                     "Please run 'grid-proxy-init -rfc' and try again.")

    # use X509_USER_PROXY from environment if set
    if os.environ.has_key('X509_USER_PROXY'):
        filePath = os.environ['X509_USER_PROXY']
        if validate_proxy(filePath):
            return filePath, filePath
        else:
            raise RuntimeError(rfc_proxy_msg)

    # use X509_USER_CERT and X509_USER_KEY if set
    if (os.environ.has_key('X509_USER_CERT') and
        os.environ.has_key('X509_USER_KEY')):
        certFile = os.environ['X509_USER_CERT']
        keyFile = os.environ['X509_USER_KEY']
        return certFile, keyFile

    # search for proxy file on disk
    uid = os.getuid()
    path = "/tmp/x509up_u%d" % uid

    if os.access(path, os.R_OK):
        if validate_proxy(path):
            return path, path
        else:
            raise RuntimeError(rfc_proxy_msg)

    # if we get here could not find a credential
    raise RuntimeError(rfc_proxy_msg)
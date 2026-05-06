def tls(force=False):
    """
    Middleware implementing TLS for SMTP connections. By
    default this is not forced- TLS is only used if
    STARTTLS is available. If the *force* parameter is set
    to True, it will not query the server for TLS features
    before upgrading to TLS.
    """
    def middleware(conn):
        if force or conn.has_extn('STARTTLS'):
            conn.starttls()
            conn.ehlo()
    return middleware
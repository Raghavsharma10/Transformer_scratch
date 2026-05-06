def _validate_conn(self, conn):
        """
        Called right before a request is made, after the socket is created.
        """
        super(HTTPSConnectionPool, self)._validate_conn(conn)

        # Force connect early to allow us to validate the connection.
        if not getattr(conn, 'sock', None):  # AppEngine might not have  `.sock`
            conn.connect()

        if not conn.is_verified:
            warnings.warn((
                'Unverified HTTPS request is being made. '
                'Adding certificate verification is strongly advised. See: '
                'https://urllib3.readthedocs.org/en/latest/security.html'),
                InsecureRequestWarning)
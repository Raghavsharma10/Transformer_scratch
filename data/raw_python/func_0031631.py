def parse_connection_string(self, connection):
        """Parse string as returned by the ``connected_users_info`` or ``user_sessions_info`` API calls.

        >>> EjabberdBackendBase().parse_connection_string('c2s_tls')
        (0, True, False)
        >>> EjabberdBackendBase().parse_connection_string('c2s_compressed_tls')
        (0, True, True)
        >>> EjabberdBackendBase().parse_connection_string('http_bind')
        (2, None, None)

        :param connection: The connection string as returned by the ejabberd APIs.
        :type  connection: str
        :return: A tuple representing the conntion type, if it is encrypted and if it uses XMPP stream
            compression.
        :rtype: tuple
        """
        # TODO: Websockets, HTTP Polling
        if connection == 'c2s_tls':
            return CONNECTION_XMPP, True, False
        elif connection == 'c2s_compressed_tls':
            return CONNECTION_XMPP, True, True
        elif connection == 'http_bind':
            return CONNECTION_HTTP_BINDING, None, None
        elif connection == 'c2s':
            return CONNECTION_XMPP, False, False
        log.warn('Could not parse connection string "%s"', connection)
        return CONNECTION_UNKNOWN, True, True
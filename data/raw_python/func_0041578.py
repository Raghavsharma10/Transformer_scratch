def _set_conn(self):
        """Establish connection to the server"""
        if self._tls:
            ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_NEVER)
        try:
            conn = ldap.initialize(self._url)
            conn.set_option(ldap.OPT_NETWORK_TIMEOUT, self._timeout)
            conn.simple_bind_s(self._binddn, self._bindpw)
        except Exception as e:
            if hasattr(e, 'message') and 'desc' in e.message:
                msg = e.message['desc']
            else:
                msg = e.args[0]['desc']
            log.critical(msg)
            raise
        log.debug('%s connection established' % ('LDAPS' if self._tls else 'LDAP'))
        self._conn = conn
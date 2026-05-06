def _set_fqdn(self):
        """Get FQDN from LDAP"""
        results = self._search(
            'cn=config',
            '(objectClass=*)',
            ['nsslapd-localhost'],
            scope=ldap.SCOPE_BASE
        )
        if not results and type(results) is not list:
            r = None
        else:
            dn, attrs = results[0]
            r = attrs['nsslapd-localhost'][0].decode('utf-8')
        self._fqdn = r
        log.debug('FQDN: %s' % self._fqdn)
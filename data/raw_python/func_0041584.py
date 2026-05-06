def _set_base_dn(self):
        """Get Base DN from LDAP"""
        results = self._search(
            'cn=config',
            '(objectClass=*)',
            ['nsslapd-defaultnamingcontext'],
            scope=ldap.SCOPE_BASE
        )
        if results and type(results) is list:
            dn, attrs = results[0]
            r = attrs['nsslapd-defaultnamingcontext'][0].decode('utf-8')
        else:
            raise Exception
        self._base_dn = r
        self._active_user_base = 'cn=users,cn=accounts,' + self._base_dn
        self._stage_user_base = 'cn=staged users,cn=accounts,cn=provisioning,' + self._base_dn
        self._preserved_user_base = 'cn=deleted users,cn=accounts,cn=provisioning,' + self._base_dn
        self._groups_base = 'cn=groups,cn=accounts,' + self._base_dn
        log.debug('Base DN: %s' % self._base_dn)
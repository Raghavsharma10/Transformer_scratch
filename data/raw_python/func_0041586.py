def _get_users(self, user_base):
        """"Get users from LDAP"""
        results = self._search(
            getattr(self, '_%s_user_base' % user_base),
            '(objectClass=*)',
            ['*'],
            scope=ldap.SCOPE_ONELEVEL
        )
        for dn, attrs in results:
            uid = attrs.get('uid')[0].decode('utf-8', 'ignore')
            getattr(self, '_%s_users' % user_base)[uid] = FreeIPAUser(dn, attrs)
            # print(attrs)
        log.debug('%s users: %s' % (user_base.capitalize(), len(getattr(self, '_%s_users' % user_base))))
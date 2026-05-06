def _search(self, base, fltr, attrs=None, scope=ldap.SCOPE_SUBTREE):
        """Perform LDAP search"""
        try:
            results = self._conn.search_s(base, scope, fltr, attrs)
        except Exception as e:
            log.exception(self._get_ldap_msg(e))
            results = False
        return results
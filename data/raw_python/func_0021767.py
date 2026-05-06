def search(self, filter, base_dn=None, attrs=None, scope=None,
               timeout=None, limit=None):
        """
        Search the directory.
        """
        if base_dn is None:
            base_dn = self._search_defaults.get('base_dn', '')
        if attrs is None:
            attrs = self._search_defaults.get('attrs', None)
        if scope is None:
            scope = self._search_defaults.get('scope', ldap.SCOPE_SUBTREE)
        if timeout is None:
            timeout = self._search_defaults.get('timeout', -1)
        if limit is None:
            limit = self._search_defaults.get('limit', 0)

        results = self.connection.search_ext_s(
            base_dn, scope, filter, attrs, timeout=timeout, sizelimit=limit)
        return self.to_items(results)
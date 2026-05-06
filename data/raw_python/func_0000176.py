def _cache_get_for_dn(self, dn: str) -> Dict[str, bytes]:
        """
        Object state is cached. When an update is required the update will be
        simulated on this cache, so that rollback information can be correct.
        This function retrieves the cached data.
        """

        # no cached item, retrieve from ldap
        self._do_with_retry(
            lambda obj: obj.search(
                dn,
                '(objectclass=*)',
                ldap3.BASE,
                attributes=['*', '+']))
        results = self._obj.response
        if len(results) < 1:
            raise NoSuchObject("No results finding current value")
        if len(results) > 1:
            raise RuntimeError("Too many results finding current value")

        return results[0]['raw_attributes']
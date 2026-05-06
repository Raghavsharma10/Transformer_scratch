def target_key(self, key):
        """Temporarily retarget the client for one call to route
        specifically to the one host that the given key routes to.  In
        that case the result on the promise is just the one host's value
        instead of a dictionary.

        .. versionadded:: 1.3
        """
        router = self.connection_pool.cluster.get_router()
        host_id = router.get_host_for_key(key)
        rv = self.target([host_id])
        rv.__resolve_singular_result = True
        return rv
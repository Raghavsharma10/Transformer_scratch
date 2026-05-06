def remove_host(self, host_id):
        """Removes a host from the client.  This only really useful for
        unittests.
        """
        with self._lock:
            rv = self._hosts.pop(host_id, None) is not None
            pool = self._pools.pop(host_id, None)
            if pool is not None:
                pool.disconnect()
            self._hosts_age += 1
            return rv
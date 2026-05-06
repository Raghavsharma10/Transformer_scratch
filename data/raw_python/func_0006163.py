def disconnect_pools(self):
        """Disconnects all connections from the internal pools."""
        with self._lock:
            for pool in self._pools.itervalues():
                pool.disconnect()
            self._pools.clear()
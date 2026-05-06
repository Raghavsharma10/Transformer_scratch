def release(self, conn):
        """Release a previously acquired connection.
        The connection is put back into the pool."""
        self._pool_lock.acquire()
        self._pool.put(ConnectionWrapper(self._pool, conn))
        self._current_acquired -= 1
        self._pool_lock.release()
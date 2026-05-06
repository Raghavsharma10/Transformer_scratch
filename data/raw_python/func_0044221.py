def acquire(self, timeout=None):
        """Acquire a connection
        :param timeout: If provided, seconds to wait for a connection before raising
            Queue.Empty. If not provided, blocks indefinitely.
        :returns: Returns a RethinkDB connection
        :raises Empty: No resources are available before timeout.
        """
        self._pool_lock.acquire()
        if timeout is None:
            conn_wrapper = self._pool.get_nowait()
        else:
            conn_wrapper = self._pool.get(True, timeout)
        self._current_acquired += 1
        self._pool_lock.release()
        return conn_wrapper.connection
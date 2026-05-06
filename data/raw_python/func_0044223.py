def release_pool(self):
        """Release pool and all its connection"""
        if self._current_acquired > 0:
            raise PoolException("Can't release pool: %d connection(s) still acquired" % self._current_acquired)
        while not self._pool.empty():
            conn = self.acquire()
            conn.close()
        if self._cleanup_thread is not None:
            self._thread_event.set()
            self._cleanup_thread.join()
        self._pool = None
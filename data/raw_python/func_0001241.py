def _write_lock(self):
        """Acquire and release the lock around output calls.

        This should allow multiple threads or processes to write output
        reliably.  Code that modifies the `_content` attribute should also do
        so within this context.
        """
        if self._lock:
            lgr.debug("Acquiring write lock")
            self._lock.acquire()
        try:
            yield
        finally:
            if self._lock:
                lgr.debug("Releasing write lock")
                self._lock.release()
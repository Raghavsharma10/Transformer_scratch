def _link(self, next_worker, next_is_first=False):
        """Link the worker to the given next worker object, 
        connecting the two workers with communication tubes."""

        lock = multiprocessing.Lock()
        next_worker._lock_prev_input = lock
        self._lock_next_input = lock
        lock.acquire()

        lock = multiprocessing.Lock()
        next_worker._lock_prev_output = lock
        self._lock_next_output = lock
        lock.acquire()

        # If the next worker is the first one, trigger it now.
        if next_is_first:
            self._lock_next_input.release()
            self._lock_next_output.release()
def putResult(self, result):
        """Register the *result* by putting it on all the output tubes."""
        self._lock_prev_output.acquire()
        for tube in self._tubes_result_output:
            tube.put((result, 0))
        self._lock_next_output.release()
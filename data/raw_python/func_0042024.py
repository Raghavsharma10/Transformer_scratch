def result(self, timeout=None):
        """
        Waits up to timeout for the result the threaded job.
        Returns immediately the result if the job has already been done.

        :param timeout: The maximum time to wait for a result (in seconds)
        :raise OSError: The timeout raised before the job finished
        :raise Exception: Raises the exception that occurred executing
                          the method
        """
        if self._done_event.wait(timeout) or self._done_event.is_set():
            if self._exception is not None:
                raise self._exception

            return self._result

        raise OSError("Timeout raised")
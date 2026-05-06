def expect(self, searcher, timeout=3):
        """Wait for input matching *searcher*

        Waits for input matching *searcher* for up to *timeout* seconds. If
        a match is found, the match result is returned (the specific type of
        returned result depends on the :class:`Searcher` type). If no match is
        found within *timeout* seconds, raise an :class:`ExpectTimeout`
        exception.

        :param Searcher searcher: :class:`Searcher` to apply to underlying
            stream.
        :param float timeout: Timeout in seconds.
        """
        timeout = float(timeout)
        end = time.time() + timeout
        match = searcher.search(self._history[self._start:])
        while not match:
            # poll() will raise ExpectTimeout if time is exceeded
            incoming = self._stream_adapter.poll(end - time.time())
            self.input_callback(incoming)
            self._history += incoming
            match = searcher.search(self._history[self._start:])
            trimlength = len(self._history) - self._window
            if trimlength > 0:
                self._start -= trimlength
                self._history = self._history[trimlength:]

        self._start += match.end
        if (self._start < 0):
            self._start = 0

        return match
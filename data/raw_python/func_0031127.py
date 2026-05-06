def poll(self, timeout):
        """
        :param float timeout: Timeout in seconds.
        """
        timeout = float(timeout)
        end_time = time.time() + timeout
        while True:
            # Keep reading until data is received or timeout
            incoming = self.stream.read(self._max_read)
            if incoming:
                return incoming
            if (end_time - time.time()) < 0:
                raise ExpectTimeout()
            time.sleep(self._poll_period)
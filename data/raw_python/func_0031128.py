def poll(self, timeout):
        """
        :param float timeout: Timeout in seconds. A timeout that is less than
            the poll_period will still cause a single read that may take up to
            poll_period seconds.
        """
        now = time.time()
        end_time = now + float(timeout)
        prev_timeout = self.stream.gettimeout()
        self.stream.settimeout(self._poll_period)
        incoming = None
        try:
            while (end_time - now) >= 0:
                try:
                    incoming = self.stream.recv(self._max_read)
                except socket.timeout:
                    pass
                if incoming:
                    return incoming
                now = time.time()
            raise ExpectTimeout()
        finally:
            self.stream.settimeout(prev_timeout)
def finish(self):
        """
        Finish this connection.
        """
        assert self._request, "Request closed"
        self._request_finished = True
        if self.m2req.should_close() or self.no_keep_alive:
            self._send("")
        self._request = None
def _fetch(self):
        """ Receive at least one message from the server, if available.

        :return: 2-tuple of number of detail messages and number of summary messages fetched
        """
        if self.closed():
            raise self.Error("Failed to read from closed connection {!r}".format(self.server.address))
        if self.defunct():
            raise self.Error("Failed to read from defunct connection {!r}".format(self.server.address))
        if not self.responses:
            return 0, 0

        self._receive()

        details, summary_signature, summary_metadata = self._unpack()

        if details:
            log_debug("[#%04X]  S: RECORD * %d", self.local_port, len(details))  # TODO
            self.responses[0].on_records(details)

        if summary_signature is None:
            return len(details), 0

        response = self.responses.popleft()
        response.complete = True
        if summary_signature == b"\x70":
            log_debug("[#%04X]  S: SUCCESS %r", self.local_port, summary_metadata)
            response.on_success(summary_metadata or {})
        elif summary_signature == b"\x7E":
            self._last_run_statement = None
            log_debug("[#%04X]  S: IGNORED", self.local_port)
            response.on_ignored(summary_metadata or {})
        elif summary_signature == b"\x7F":
            self._last_run_statement = None
            log_debug("[#%04X]  S: FAILURE %r", self.local_port, summary_metadata)
            response.on_failure(summary_metadata or {})
        else:
            self._last_run_statement = None
            raise ProtocolError("Unexpected response message with signature %02X" % summary_signature)

        return len(details), 1
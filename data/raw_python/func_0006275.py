def start_event_stream(self):
        """ Start streaming events from `gerrit stream-events`. """
        if not self._stream:
            self._stream = GerritStream(self, ssh_client=self._ssh_client)
            self._stream.start()
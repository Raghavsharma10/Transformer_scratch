def handle_PoisonPillFrame(self, frame):
        """ Is sent in case protocol lost connection to server."""
        # Will be delivered after Close or CloseOK handlers. It's for channels,
        # so ignore it.
        if self.connection.closed.done():
            return
        # If connection was not closed already - we lost connection.
        # Protocol should already be closed
        self._close_all(frame.exception)
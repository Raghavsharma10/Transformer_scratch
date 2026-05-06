def close(self):
        """
        Close the channel by handshaking with the server.

        This method is a :ref:`coroutine <coroutine>`.
        """
        # If we aren't already closed ask for server to close
        if not self.is_closed():
            self._closing = True
            # Let the ChannelActor do the actual close operations.
            # It will do the work on CloseOK
            self.sender.send_Close(
                0, 'Channel closed by application', 0, 0)
            try:
                yield from self.synchroniser.wait(spec.ChannelCloseOK)
            except AMQPError:
                # For example if both sides want to close or the connection
                # is closed.
                pass
        else:
            if self._closing:
                log.warn("Called `close` on already closing channel...")
def close(self):
        """
        Close the connection by handshaking with the server.

        This method is a :ref:`coroutine <coroutine>`.
        """
        if not self.is_closed():
            self._closing = True
            # Let the ConnectionActor do the actual close operations.
            # It will do the work on CloseOK
            self.sender.send_Close(
                0, 'Connection closed by application', 0, 0)
            try:
                yield from self.synchroniser.wait(spec.ConnectionCloseOK)
            except AMQPConnectionError:
                # For example if both sides want to close or the connection
                # is closed.
                pass
        else:
            if self._closing:
                log.warn("Called `close` on already closing connection...")
        # finish all pending tasks
        yield from self.protocol.heartbeat_monitor.wait_closed()
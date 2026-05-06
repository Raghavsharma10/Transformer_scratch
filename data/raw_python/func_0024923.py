def _get_websocket(self, reuse=True):
        """
        Reuse existing connection or create a new connection.
        """
        # Check if still connected
        if self.ws and reuse:
            if self.ws.connected:
                return self.ws

            logging.debug("Stale connection, reconnecting.")

        self.ws = self._create_connection()
        return self.ws
def disconnect(self):
        """
        Disconnect from server.

        Closes the websocket, sets `ingame = False`, and emits on_sock_closed.
        """
        self.ws.close()
        self.ingame = False
        self.subscriber.on_sock_closed()
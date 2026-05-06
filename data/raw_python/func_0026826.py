def on_start(self, host, port, channel, nickname, password):
        """
        A WebSocket session has started - create a greenlet to host
        the IRC client, and start it.
        """
        self.client = WebSocketIRCClient(host, port, channel, nickname,
                                         password, self)
        self.spawn(self.client.start)
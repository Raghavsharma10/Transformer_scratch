async def connect(self):
        """Connect to gateway via SSL."""
        tcp_client = TCPTransport(self.frame_received_cb, self.connection_closed_cb)
        self.transport, _ = await self.loop.create_connection(
            lambda: tcp_client,
            host=self.config.host,
            port=self.config.port,
            ssl=self.create_ssl_context())
        self.connected = True
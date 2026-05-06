def connect(self, address, token=None):
        """
        Connect the underlying websocket to the address,
        send a handshake and optionally a token packet.

        Returns `True` if connected, `False` if the connection failed.

        :param address: string, `IP:PORT`
        :param token: unique token, required by official servers,
                      acquired through utils.find_server()
        :return: True if connected, False if not
        """
        if self.connected:
            self.subscriber.on_connect_error(
                'Already connected to "%s"' % self.address)
            return False

        self.address = address
        self.server_token = token
        self.ingame = False

        self.ws.settimeout(1)
        self.ws.connect('ws://%s' % self.address, origin='http://agar.io')
        if not self.connected:
            self.subscriber.on_connect_error(
                'Failed to connect to "%s"' % self.address)
            return False

        self.subscriber.on_sock_open()
        # allow handshake canceling
        if not self.connected:
            self.subscriber.on_connect_error(
                'Disconnected before sending handshake')
            return False

        self.send_handshake()
        if self.server_token:
            self.send_token(self.server_token)

        old_nick = self.player.nick
        self.player.reset()
        self.world.reset()
        self.player.nick = old_nick
        return True
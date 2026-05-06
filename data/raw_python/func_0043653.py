def socket(self, socket_type, identity=None, mechanism=None):
        """
        Create and register a new socket.

        :param socket_type: The type of the socket.
        :param loop: An optional event loop to associate the socket with.

        This is the preferred method to create new sockets.
        """
        socket = Socket(
            context=self,
            socket_type=socket_type,
            identity=identity,
            mechanism=mechanism,
            loop=self.loop,
        )
        self.register_child(socket)
        return socket
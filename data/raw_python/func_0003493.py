def connect(self, socket_or_address):
        """
        Connect to server and start serving registered functions.

        :type socket_or_address: tuple or socket object
        :arg  socket_or_address: A ``(host, port)`` pair to be passed
                                 to `socket.create_connection`, or
                                 a socket object.

        """
        if isinstance(socket_or_address, tuple):
            import socket
            self.socket = socket.create_connection(socket_or_address)
        else:
            self.socket = socket_or_address

        # This is what BaseServer.finish_request does:
        address = None  # it is not used, so leave it empty
        self.handler = EPCClientHandler(self.socket, address, self)

        self.call = self.handler.call
        self.call_sync = self.handler.call_sync
        self.methods = self.handler.methods
        self.methods_sync = self.handler.methods_sync

        self.handler_thread = newthread(self, target=self.handler.start)
        self.handler_thread.daemon = self.thread_daemon
        self.handler_thread.start()
        self.handler.wait_until_ready()
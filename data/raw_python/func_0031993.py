def server(self, ):
        """Creates and returns a ServerConnection

        :returns: a server connection
        :rtype: :class:`connection.ServerConnection3`
        :raises: None
        """
        c = connection.ServerConnection3(self)
        with self.mutex:
            self.connections.append(c)
        return c
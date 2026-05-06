def set_servers(self, servers):
        """
        Iter to a list of servers and instantiate Protocol class.

        :param servers: A list of servers
        :type servers: list
        :return: Returns nothing
        :rtype: None
        """
        if isinstance(servers, six.string_types):
            servers = [servers]

        assert servers, "No memcached servers supplied"
        self._servers = [Protocol(
            server=server,
            username=self.username,
            password=self.password,
            compression=self.compression,
            socket_timeout=self.socket_timeout,
            pickle_protocol=self.pickle_protocol,
            pickler=self.pickler,
            unpickler=self.unpickler,
        ) for server in servers]
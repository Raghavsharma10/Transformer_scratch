def add_client(self, client):
        """
        Adds the specified client to this manager.

        :param client: The client to add into this manager.
        :type client: :class:`revision.client.Client`
        :return: The ClientManager instance (method chaining)
        :rtype: :class:`revision.client_manager.ClientManager`
        """
        if not isinstance(client, Client):
            raise InvalidArgType()

        if self.has_client(client.key):
            return self

        self[client.key] = client

        return self
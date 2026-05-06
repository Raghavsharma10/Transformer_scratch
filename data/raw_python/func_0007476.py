def fetch_by_client_id(self, client_id):
        """
        Retrieve a client by its identifier.

        :param client_id: Identifier of a client app.
        :return: An instance of :class:`oauth2.Client`.
        :raises: ClientNotFoundError

        """
        if client_id not in self.clients:
            raise ClientNotFoundError

        return self.clients[client_id]
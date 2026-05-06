def has_commit(self, client_key=None):
        """
        Return True if client has new commit.

        :param client_key: The client key
        :type client_key: str
        :return:
        :rtype: boolean
        """
        if client_key is None and self.current_client is None:
            raise ClientNotExist()

        if client_key:
            if not self.clients.has_client(client_key):
                raise ClientNotExist()

            client = self.clients.get_client(client_key)

            return client.has_commit()

        if self.current_client:
            client = self.current_client

            return client.has_commit()

        return False
def use(self, client_key):
        """
        :param client_key: The client key.
        :type client_key: str
        :return: The Orchestrator instance (method chaining)
        :rtype: :class:`revision.orchestrator.Orchestrator`
        """
        if not self.clients.has_client(client_key):
            raise ClientNotExist()

        self.current_client = self.clients.get_client(client_key)

        return self
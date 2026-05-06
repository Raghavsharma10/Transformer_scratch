def client(self):
        """
        Helper property to lazy initialize and cache client. Runs
        :meth:`~django_docker_helpers.config.backends.base.BaseParser.get_client`.

        :return: an instance of backend-specific client
        """
        if self._client is not None:
            return self._client

        self._client = self.get_client()
        return self._client
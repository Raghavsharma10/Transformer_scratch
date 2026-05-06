def is_connected(self):
        """ Returns the connection status of the data store.

        Returns:
            bool: ``True`` if the data store is connected to the MongoDB server.
        """
        if self._client is not None:
            try:
                self._client.server_info()
            except ConnectionFailure:
                return False
            return True
        else:
            return False
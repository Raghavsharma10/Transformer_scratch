def purge(self, name=None):
        """
        Disconnect from the given database and remove from local cache

        :param name: The name of the connection
        :type name: str

        :rtype: None
        """
        self.disconnect(name)

        if name in self._connections:
            del self._connections[name]
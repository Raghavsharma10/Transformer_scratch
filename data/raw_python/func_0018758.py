def erase(self, server):
        """ Erase credentials for `server`. Raises a `StoreError` if an error
            occurs.
        """
        if not isinstance(server, six.binary_type):
            server = server.encode('utf-8')
        self._execute('erase', server)
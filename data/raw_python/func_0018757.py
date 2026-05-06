def store(self, server, username, secret):
        """ Store credentials for `server`. Raises a `StoreError` if an error
            occurs.
        """
        data_input = json.dumps({
            'ServerURL': server,
            'Username': username,
            'Secret': secret
        }).encode('utf-8')
        return self._execute('store', data_input)
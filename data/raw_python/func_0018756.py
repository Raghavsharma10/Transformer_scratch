def get(self, server):
        """ Retrieve credentials for `server`. If no credentials are found,
            a `StoreError` will be raised.
        """
        if not isinstance(server, six.binary_type):
            server = server.encode('utf-8')
        data = self._execute('get', server)
        result = json.loads(data.decode('utf-8'))

        # docker-credential-pass will return an object for inexistent servers
        # whereas other helpers will exit with returncode != 0. For
        # consistency, if no significant data is returned,
        # raise CredentialsNotFound
        if result['Username'] == '' and result['Secret'] == '':
            raise errors.CredentialsNotFound(
                'No matching credentials in {}'.format(self.program)
            )

        return result
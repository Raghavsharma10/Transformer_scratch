def retrieve_client_credentials(self):
        """Return the client credentials.

        :returns: tuple(client_id, client_secret)
        """
        client_id = self.params.get('client_id')
        client_secret = self.params.get('client_secret')
        return (client_id, client_secret)
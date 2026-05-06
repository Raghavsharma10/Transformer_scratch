def add_client(self, client_id, client_secret, redirect_uris,
                   authorized_grants=None, authorized_response_types=None):
        """
        Add a client app.

        :param client_id: Identifier of the client app.
        :param client_secret: Secret the client app uses for authentication
                              against the OAuth 2.0 provider.
        :param redirect_uris: A ``list`` of URIs to redirect to.

        """
        self.clients[client_id] = Client(
            identifier=client_id,
            secret=client_secret,
            redirect_uris=redirect_uris,
            authorized_grants=authorized_grants,
            authorized_response_types=authorized_response_types)

        return True
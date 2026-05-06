def fetch_by_client_id(self, client_id):
        """
        Retrieves a client by its identifier.

        :param client_id: The identifier of a client.

        :return: An instance of :class:`oauth2.datatype.Client`.

        :raises: :class:`oauth2.error.ClientError` if no client could be
                 retrieved.
        """
        grants = None
        redirect_uris = None
        response_types = None

        client_data = self.fetchone(self.fetch_client_query, client_id)

        if client_data is None:
            raise ClientNotFoundError

        grant_data = self.fetchall(self.fetch_grants_query, client_data[0])
        if grant_data:
            grants = []
            for grant in grant_data:
                grants.append(grant[0])

        redirect_uris_data = self.fetchall(self.fetch_redirect_uris_query,
                                           client_data[0])
        if redirect_uris_data:
            redirect_uris = []
            for redirect_uri in redirect_uris_data:
                redirect_uris.append(redirect_uri[0])

        response_types_data = self.fetchall(self.fetch_response_types_query,
                                            client_data[0])
        if response_types_data:
            response_types = []
            for response_type in response_types_data:
                response_types.append(response_type[0])

        return Client(identifier=client_data[1], secret=client_data[2],
                      authorized_grants=grants,
                      authorized_response_types=response_types,
                      redirect_uris=redirect_uris)
def fetch_existing_token_of_user(self, client_id, grant_type, user_id):
        """
        Retrieve an access token issued to a client and user for a specific
        grant.

        :param client_id: The identifier of a client as a `str`.
        :param grant_type: The type of grant.
        :param user_id: The identifier of the user the access token has been
                        issued to.

        :return: An instance of :class:`oauth2.datatype.AccessToken`.

        :raises: :class:`oauth2.error.AccessTokenNotFound` if not access token
                 could be retrieved.
        """
        token_data = self.fetchone(self.fetch_existing_token_of_user_query,
                                   client_id, grant_type, user_id)

        if token_data is None:
            raise AccessTokenNotFound

        scopes = self._fetch_scopes(access_token_id=token_data[0])

        data = self._fetch_data(access_token_id=token_data[0])

        return self._row_to_token(data=data, scopes=scopes, row=token_data)
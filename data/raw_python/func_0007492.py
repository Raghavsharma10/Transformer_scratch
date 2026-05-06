def fetch_by_refresh_token(self, refresh_token):
        """
        Retrieves an access token by its refresh token.

        :param refresh_token: The refresh token of an access token as a `str`.

        :return: An instance of :class:`oauth2.datatype.AccessToken`.

        :raises: :class:`oauth2.error.AccessTokenNotFound` if not access token
                 could be retrieved.
        """
        row = self.fetchone(self.fetch_by_refresh_token_query, refresh_token)

        if row is None:
            raise AccessTokenNotFound

        scopes = self._fetch_scopes(access_token_id=row[0])

        data = self._fetch_data(access_token_id=row[0])

        return self._row_to_token(data=data, scopes=scopes, row=row)
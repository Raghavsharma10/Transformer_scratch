def save_token(self, access_token):
        """
        Stores an access token and additional data in memory.

        :param access_token: An instance of :class:`oauth2.datatype.AccessToken`.
        """
        self.access_tokens[access_token.token] = access_token

        unique_token_key = self._unique_token_key(access_token.client_id,
                                                  access_token.grant_type,
                                                  access_token.user_id)

        self.unique_token_identifier[unique_token_key] = access_token.token

        if access_token.refresh_token is not None:
            self.refresh_tokens[access_token.refresh_token] = access_token

        return True
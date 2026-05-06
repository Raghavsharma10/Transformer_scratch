def save_token(self, access_token):
        """
        Stores the access token and additional data in redis.

        See :class:`oauth2.store.AccessTokenStore`.

        """
        self.write(access_token.token, access_token.__dict__)

        unique_token_key = self._unique_token_key(access_token.client_id,
                                                  access_token.grant_type,
                                                  access_token.user_id)
        self.write(unique_token_key, access_token.__dict__)

        if access_token.refresh_token is not None:
            self.write(access_token.refresh_token, access_token.__dict__)
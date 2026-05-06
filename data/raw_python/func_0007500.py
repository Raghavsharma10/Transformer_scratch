def save_token(self, access_token):
        """
        Stores the access token and additional data in memcache.

        See :class:`oauth2.store.AccessTokenStore`.

        """
        key = self._generate_cache_key(access_token.token)
        self.mc.set(key, access_token.__dict__)

        unique_token_key = self._unique_token_key(access_token.client_id,
                                                  access_token.grant_type,
                                                  access_token.user_id)
        self.mc.set(self._generate_cache_key(unique_token_key),
                    access_token.__dict__)

        if access_token.refresh_token is not None:
            rft_key = self._generate_cache_key(access_token.refresh_token)
            self.mc.set(rft_key, access_token.__dict__)
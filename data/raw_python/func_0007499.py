def save_code(self, authorization_code):
        """
        Stores the data belonging to an authorization code token in memcache.

        See :class:`oauth2.store.AuthCodeStore`.

        """
        key = self._generate_cache_key(authorization_code.code)

        self.mc.set(key, {"client_id": authorization_code.client_id,
                          "code": authorization_code.code,
                          "expires_at": authorization_code.expires_at,
                          "redirect_uri": authorization_code.redirect_uri,
                          "scopes": authorization_code.scopes,
                          "data": authorization_code.data,
                          "user_id": authorization_code.user_id})
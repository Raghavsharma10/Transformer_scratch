def save_code(self, authorization_code):
        """
        Stores the data belonging to an authorization code token in redis.

        See :class:`oauth2.store.AuthCodeStore`.

        """
        self.write(authorization_code.code,
                   {"client_id": authorization_code.client_id,
                    "code": authorization_code.code,
                    "expires_at": authorization_code.expires_at,
                    "redirect_uri": authorization_code.redirect_uri,
                    "scopes": authorization_code.scopes,
                    "data": authorization_code.data,
                    "user_id": authorization_code.user_id})
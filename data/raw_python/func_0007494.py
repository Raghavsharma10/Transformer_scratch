def save_token(self, access_token):
        """
        Creates a new entry for an access token in the database.

        :param access_token: An instance of
                             :class:`oauth2.datatype.AccessToken`.

         :return: `True`.
        """
        access_token_id = self.execute(self.create_access_token_query,
                                       access_token.client_id,
                                       access_token.grant_type,
                                       access_token.token,
                                       access_token.expires_at,
                                       access_token.refresh_token,
                                       access_token.refresh_expires_at,
                                       access_token.user_id)

        for key, value in list(access_token.data.items()):
            self.execute(self.create_data_query, key, value,
                         access_token_id)

        for scope in access_token.scopes:
            self.execute(self.create_scope_query, scope, access_token_id)

        return True
def save_code(self, authorization_code):
        """
        Creates a new entry of an auth code in the database.

        :param authorization_code: An instance of
                                   :class:`oauth2.datatype.AuthorizationCode`.

        :return: `True` if everything went fine.
        """
        auth_code_id = self.execute(self.create_auth_code_query,
                                    authorization_code.client_id,
                                    authorization_code.code,
                                    authorization_code.expires_at,
                                    authorization_code.redirect_uri,
                                    authorization_code.user_id)

        for key, value in list(authorization_code.data.items()):
            self.execute(self.create_data_query, key, value, auth_code_id)

        for scope in authorization_code.scopes:
            self.execute(self.create_scope_query, scope, auth_code_id)

        return True
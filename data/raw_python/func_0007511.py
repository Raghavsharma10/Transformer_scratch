def process(self, request, response, environ):
        """
        Generates a new access token and returns it.

        Returns the access token and the type of the token as JSON.

        Calls `oauth2.store.AccessTokenStore` to persist the token.
        """
        token_data = self.create_token(
            client_id=self.client.identifier,
            data=self.data,
            grant_type=AuthorizationCodeGrant.grant_type,
            scopes=self.scopes,
            user_id=self.user_id)

        self.auth_code_store.delete_code(self.code)

        if self.scopes:
            token_data["scope"] = encode_scopes(self.scopes)

        json_success_response(data=token_data, response=response)

        return response
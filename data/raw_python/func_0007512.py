def process(self, request, response, environ):
        """
        Takes the incoming request, asks the concrete SiteAdapter to validate
        it and issues a new access token that is returned to the client on
        successful validation.
        """
        try:
            data = self.site_adapter.authenticate(request, environ,
                                                  self.scope_handler.scopes,
                                                  self.client)
            data = AuthorizeMixin.sanitize_return_value(data)
        except UserNotAuthenticated:
            raise OAuthInvalidError(error="invalid_client",
                                    explanation=self.OWNER_NOT_AUTHENTICATED)

        if isinstance(data, Response):
            return data

        token_data = self.create_token(
            client_id=self.client.identifier,
            data=data[0],
            grant_type=ResourceOwnerGrant.grant_type,
            scopes=self.scope_handler.scopes,
            user_id=data[1])

        if self.scope_handler.send_back:
            token_data["scope"] = encode_scopes(self.scope_handler.scopes)

        json_success_response(data=token_data, response=response)

        return response
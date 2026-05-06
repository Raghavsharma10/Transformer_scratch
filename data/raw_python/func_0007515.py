def read_validate_params(self, request):
        """
        Validate the incoming request.

        :param request: The incoming :class:`oauth2.web.Request`.

        :return: Returns ``True`` if data is valid.

        :raises: :class:`oauth2.error.OAuthInvalidError`

        """
        self.refresh_token = request.post_param("refresh_token")

        if self.refresh_token is None:
            raise OAuthInvalidError(
                error="invalid_request",
                explanation="Missing refresh_token in request body")

        self.client = self.client_authenticator.by_identifier_secret(request)

        try:
            access_token = self.access_token_store.fetch_by_refresh_token(
                self.refresh_token
            )
        except AccessTokenNotFound:
            raise OAuthInvalidError(error="invalid_request",
                                    explanation="Invalid refresh token")

        refresh_token_expires_at = access_token.refresh_expires_at
        self.refresh_grant_type = access_token.grant_type

        if refresh_token_expires_at != 0 and \
                        refresh_token_expires_at < int(time.time()):
            raise OAuthInvalidError(error="invalid_request",
                                    explanation="Invalid refresh token")

        self.data = access_token.data
        self.user_id = access_token.user_id

        self.scope_handler.parse(request, "body")
        self.scope_handler.compare(access_token.scopes)

        return True
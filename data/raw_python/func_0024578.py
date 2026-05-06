def token(self):
        """
        Returns authorization token provided by Cocaine.

        The real meaning of the token is determined by its type. For example OAUTH2 token will
        have "bearer" type.

        :return: A tuple of token type and body.
        """
        if self._token is None:
            token_type = os.getenv(TOKEN_TYPE_KEY, '')
            token_body = os.getenv(TOKEN_BODY_KEY, '')
            self._token = _Token(token_type, token_body)
        return self._token
def authorize(self, code):
        """Obtain and set authorization tokens based on ``code``.

        :param code: The code obtained by an out-of-band authorization request
            to Reddit.

        """
        if self._authenticator.redirect_uri is None:
            raise InvalidInvocation("redirect URI not provided")
        self._request_token(
            code=code,
            grant_type="authorization_code",
            redirect_uri=self._authenticator.redirect_uri,
        )
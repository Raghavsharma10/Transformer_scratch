def claim_token(self, **params):
        """Claim current token by POSTing 'login' and 'password'.

        User's `Authorization` header value is returned in `WWW-Authenticate`
        header.
        """
        self._json_params.update(params)
        success, self.user = self.Model.authenticate_by_password(
            self._json_params)

        if success:
            headers = remember(self.request, self.user.username)
            return JHTTPOk('Token claimed', headers=headers)
        if self.user:
            raise JHTTPUnauthorized('Wrong login or password')
        else:
            raise JHTTPNotFound('User not found')
def reset_token(self, **params):
        """ Reset current token by POSTing 'login' and 'password'.

        User's `Authorization` header value is returned in `WWW-Authenticate`
        header.
        """
        response = self.claim_token(**params)
        if not self.user:
            return response

        self.user.api_key.reset_token()
        headers = remember(self.request, self.user.username)
        return JHTTPOk('Registered', headers=headers)
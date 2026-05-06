def refresh(self):
        """Obtain a new access token from the refresh_token."""
        if self.refresh_token is None:
            raise InvalidInvocation("refresh token not provided")
        self._request_token(
            grant_type="refresh_token", refresh_token=self.refresh_token
        )
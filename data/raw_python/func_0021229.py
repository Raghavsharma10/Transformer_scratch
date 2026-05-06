def refresh(self):
        """Obtain a new personal-use script type access token."""
        self._request_token(
            grant_type="password",
            username=self._username,
            password=self._password,
        )
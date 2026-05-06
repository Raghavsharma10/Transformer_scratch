def get_token(self):
        """
        get current oauth token
        """
        self.token = self._session.fetch_token(
            token_url=CLOUD_URLS["get_token"][1],
            client_id=self._client_id,
            client_secret=self._client_secret
        )
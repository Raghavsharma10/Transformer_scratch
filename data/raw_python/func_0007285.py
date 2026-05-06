def _refresh_oath_token(self):
        """
        Refresh Monzo OAuth 2 token.

        Official docs:
            https://monzo.com/docs/#refreshing-access

        :raises UnableToRefreshTokenException: when token couldn't be refreshed
        """
        url = urljoin(self.api_url, '/oauth2/token')
        data = {
            'grant_type': 'refresh_token',
            'client_id': self._client_id,
            'client_secret': self._client_secret,
            'refresh_token': self._token['refresh_token'],
        }

        token_response = requests.post(url, data=data)
        token = token_response.json()

        # Not ideal, but that's how Monzo API returns errors
        if 'error' in token:
            raise CantRefreshTokenError(
                "Unable to refresh the token: {}".format(token)
            )

        self._token = token
        self._save_token_on_disk()
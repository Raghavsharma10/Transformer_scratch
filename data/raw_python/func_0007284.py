def _get_oauth_token(self):
        """
        Get Monzo access token via OAuth2 `authorization code` grant type.

        Official docs:
            https://monzo.com/docs/#acquire-an-access-token

        :returns: OAuth 2 access token
        :rtype: dict
        """
        url = urljoin(self.api_url, '/oauth2/token')

        oauth = OAuth2Session(
            client_id=self._client_id,
            redirect_uri=config.REDIRECT_URI,
        )

        token = oauth.fetch_token(
            token_url=url,
            code=self._auth_code,
            client_secret=self._client_secret,
        )

        return token
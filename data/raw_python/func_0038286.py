def obtain_token(self, redirect_url: str, state: str) -> str:
        """
        Exchange the code that was obtained using `authorize_url` for an authorization token. The code is extracted
        from the URL that redirected the user back to your site.

        Example:
            >>> auth = OAuthAuthentication('https://example.com/oauth/moneybird/', 'your_id', 'your_secret')
            >>> auth.obtain_token('https://example.com/oauth/moneybird/?code=any&state=random_string', 'random_string')
            'token_for_auth'
            >>> auth.is_ready()
            True

        :param redirect_url: The full URL the user was redirected to.
        :param state: The state used in the authorize url.
        :return: The authorization token.
        """
        url_data = parse_qs(redirect_url.split('?', 1)[1])

        if 'error' in url_data:
            logger.warning("Error received in OAuth authentication response: %s" % url_data.get('error'))
            raise OAuthAuthentication.OAuthError(url_data['error'], url_data.get('error_description', None))

        if 'code' not in url_data:
            logger.error("The provided URL is not a valid OAuth authentication response: no code")
            raise ValueError("The provided URL is not a valid OAuth authentication response: no code")

        if state and [state] != url_data['state']:
            logger.warning("OAuth CSRF attack detected: the state in the provided URL does not equal the given state")
            raise ValueError("CSRF attack detected: the state in the provided URL does not equal the given state")

        try:
            response = requests.post(
                url=urljoin(self.base_url, self.token_url),
                data={
                    'grant_type': 'authorization_code',
                    'code': url_data['code'][0],
                    'redirect_uri': self.redirect_url,
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                },
            ).json()
        except ValueError:
            logger.error("The OAuth server returned an invalid response when obtaining a token: JSON error")
            raise ValueError("The OAuth server returned an invalid response when obtaining a token: JSON error")

        if 'error' in response:
            logger.warning("Error while obtaining OAuth authorization token: %s" % response['error'])
            raise OAuthAuthentication.OAuthError(response['error'], response.get('error', ''))

        if 'access_token' not in response:
            logger.error("The OAuth server returned an invalid response when obtaining a token: no access token")
            raise ValueError("The remote server returned an invalid response when obtaining a token: no access token")

        self.real_auth.set_token(response['access_token'])
        logger.debug("Obtained authentication token for state %s: %s" % (state, self.real_auth.auth_token))

        return response['access_token']
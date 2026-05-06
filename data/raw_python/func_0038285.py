def authorize_url(self, scope: list, state: str = None) -> tuple:
        """
        Returns the URL to which the user can be redirected to authorize your application to access his/her account. It
        will also return the state which can be used for CSRF protection. A state is generated if not passed to this
        method.

        Example:
            >>> auth = OAuthAuthentication('https://example.com/oauth/moneybird/', 'your_id', 'your_secret')
            >>> auth.authorize_url()
            ('https://moneybird.com/oauth/authorize?client_id=your_id&redirect_uri=https%3A%2F%2Fexample.com%2Flogin%2F
            moneybird&state=random_string', 'random_string')

        :param scope: The requested scope.
        :param state: Optional state, when omitted a random value is generated.
        :return: 2-tuple containing the URL to redirect the user to and the randomly generated state.
        """
        url = urljoin(self.base_url, self.auth_url)
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_url,
            'scope': ' '.join(scope),
            'state': state if state is not None else self._generate_state(),
        }

        return "%s?%s" % (url, urlencode(params)), params['state']
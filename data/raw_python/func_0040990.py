def access_token(self):
        """
        Retrieve and cache an access token to authenticate API calls.
        :return: An access token string.
        """
        if self._cached_access_token is not None:
            return self._cached_access_token
        resp = self._request(endpoint='access_token', data={'grant_type': 'client_credentials', 'scope': 'basic user'},
                             auth=(self.api_username, self.api_key))
        self._cached_access_token = resp['access_token']
        return self._cached_access_token
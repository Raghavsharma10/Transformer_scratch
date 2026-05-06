def exchange_code(self, code):
        """Exchange one-use code for an access_token and request_token."""
        params = {'client_id': self.client_id,
                  'client_secret': self.client_secret,
                  'grant_type': 'authorization_code',
                  'code': code}
        result = self._send_request(EXCHANGE_URL.format(self._base_url),
                                    params=params, method='POST',
                                    data_field=None)
        self.access_token = result['access_token']
        self.refresh_token = result['refresh_token']
        return self.access_token, self.refresh_token
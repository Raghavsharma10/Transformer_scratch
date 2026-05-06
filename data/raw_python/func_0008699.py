def exchange_pin(self, pin):
        """Exchange one-use pin for an access_token and request_token."""
        params = {'client_id': self.client_id,
                  'client_secret': self.client_secret,
                  'grant_type': 'pin',
                  'pin': pin}
        result = self._send_request(EXCHANGE_URL.format(self._base_url),
                                    params=params, method='POST',
                                    data_field=None)
        self.access_token = result['access_token']
        self.refresh_token = result['refresh_token']
        return self.access_token, self.refresh_token
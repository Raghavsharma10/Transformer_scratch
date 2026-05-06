def refresh_access_token(self):
        """
        Refresh the access_token.

        The self.access_token attribute will be updated with the value of the
        new access_token which will also be returned.
        """
        if self.client_secret is None:
            raise Exception("client_secret must be set to execute "
                            "refresh_access_token.")
        if self.refresh_token is None:
            raise Exception("refresh_token must be set to execute "
                            "refresh_access_token.")
        params = {'client_id': self.client_id,
                  'client_secret': self.client_secret,
                  'grant_type': 'refresh_token',
                  'refresh_token': self.refresh_token}
        result = self._send_request(REFRESH_URL.format(self._base_url),
                                    params=params, method='POST',
                                    data_field=None)
        self.access_token = result['access_token']
        return self.access_token
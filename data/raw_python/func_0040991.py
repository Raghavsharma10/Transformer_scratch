def _call(self, endpoint, data=None):
        """
        Make an authorized API call to specified endpoint.
        :param str endpoint: API endpoint's relative URL, eg. `/account`.
        :param dict data: POST request data.
        :return: A dictionary or a string with response data.
        """
        data = {} if data is None else data
        try:
            data['access_token'] = self.access_token()
            return self._request(endpoint, data)
        except AccessTokenExpired:
            self._cached_access_token = None
            data['access_token'] = self.access_token()
            return self._request(endpoint, data)
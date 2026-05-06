def _request(self, endpoint, data, auth=None):
        """
        Make HTTP POST request to an API endpoint.
        :param str endpoint: API endpoint's relative URL, eg. `/account`.
        :param dict data: POST request data.
        :param tuple auth: HTTP basic auth credentials.
        :return: A dictionary or a string with response data.
        """
        url = '{}/{}'.format(self.base_url, endpoint)
        response = requests.post(url, data, auth=auth)
        return self._handle_response(response)
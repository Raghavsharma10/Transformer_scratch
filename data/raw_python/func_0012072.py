def request(self, endpoint, method="GET", params=None):
        """Return dict of response received from Safecast's API
        :param endpoint: (required) Full url or Safecast API endpoint
                         (e.g. measurements/users)
        :type endpoint: string
        :param method: (optional) Method of accessing data, either
                       GET, POST, PUT or DELETE. (default GET)
        :type method: string
        :param params: (optional) Dict of parameters (if any) accepted
                       the by Safecast API endpoint you are trying to
                       access (default None)
        :type params: dict or None
        :rtype: dict
        """

        # In case they want to pass a full Safecast URL
        # i.e. https://api.safecast.org/measurements.json
        if endpoint.startswith("http"):
            url = endpoint
        else:
            url = "%s/%s.json" % (self.api_url, endpoint)

        if method != "GET":
            if self.api_key is None:
                raise SafecastPyAuthError("Require an api_key")
            url = url + "?api_key={0}".format(self.api_key)

        content = self._request(url, method=method, params=params, api_call=url)
        return content
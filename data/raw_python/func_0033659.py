def _make_get_request(self, uri, parameters=None, timeout=None):
        """
        Given a request add in the required parameters and return the parsed
        XML object.
        """
        if not timeout:
            timeout = self.timeout
        return self._make_request(requests.get, uri, params=parameters, timeout=timeout)
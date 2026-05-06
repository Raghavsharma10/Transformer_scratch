def _make_post_request(self, uri, payload, timeout=None):
        """
        Given a request add in the required parameters and return the parsed
        XML object.
        """
        if not timeout:
            timeout = self.timeout
        return self._make_request(requests.post, uri, data=payload, timeout=timeout)
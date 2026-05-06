def _make_delete_request(self, uri, timeout=None):
        """
        Given a request add in the required parameters and return the parsed
        XML object.
        """
        if not timeout:
            timeout = self.timeout
        return self._make_request(requests.delete, uri, timeout=timeout)
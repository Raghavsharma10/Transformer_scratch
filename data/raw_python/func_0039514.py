def put(self, url, data=None):
        """Send a HTTP PUT request to a URL and return the result.
        """
        self.conn.request("PUT", url, data)
        return self._process_response()
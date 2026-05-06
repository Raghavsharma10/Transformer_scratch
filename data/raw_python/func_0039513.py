def post(self, url, data):
        """Send a HTTP POST request to a URL and return the result.
        """
        headers = {
            "Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/json"
        }
        self.conn.request("POST", url, data, headers)
        return self._process_response()
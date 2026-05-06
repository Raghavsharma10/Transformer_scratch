def data_request(self, payload, timeout=TIMEOUT):
        """Perform a data_request and return the result."""
        request_url = self.base_url + "/data_request"
        return requests.get(request_url, timeout=timeout, params=payload)
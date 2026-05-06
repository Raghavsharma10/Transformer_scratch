def post_request(self, endpoint, body=None, timeout=-1):
        """
        Perform a POST request to a given endpoint in UpCloud's API.
        """
        return self.request('POST', endpoint, body, timeout)
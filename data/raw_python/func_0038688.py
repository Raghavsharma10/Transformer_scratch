def request(self, action, data={}, headers={}, method='GET'):
        """
        Append the REST headers to every request
        """
        headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "X-Version": "1",
            "Accept": "application/json"
        }

        return Transport.request(self, action, data, headers, method)
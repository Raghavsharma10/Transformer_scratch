def post(self, action, data=None, headers=None):
        """Makes a GET request
        """
        return self.request(make_url(self.endpoint, action), method='POST', data=data,
                            headers=headers)
def get(self, action, params=None, headers=None):
        """Makes a GET request
        """
        return self.request(make_url(self.endpoint, action), method='GET', data=params,
                            headers=headers)
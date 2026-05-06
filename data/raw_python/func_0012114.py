def delete(self, action, headers=None):
        """Makes a GET request
        """
        return self.request(make_url(self.endpoint, action), method='DELETE',
                            headers=headers)
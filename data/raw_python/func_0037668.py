def request(self, action, data={}, headers={}, method='GET'):
        """
        Append the user authentication details to every incoming request
        """
        data = self.merge(data, {'user': self.username, 'password': self.password, 'api_id': self.apiId})
        return Transport.request(self, action, data, headers, method)
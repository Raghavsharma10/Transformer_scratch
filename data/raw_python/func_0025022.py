def get_clients(self):
        """
        Returns the clients stored in the instance of UAA.
        """
        self.assert_has_permission('clients.read')

        uri = self.uri + '/oauth/clients'
        headers = self.get_authorization_headers()
        response = requests.get(uri, headers=headers)
        return response.json()['resources']
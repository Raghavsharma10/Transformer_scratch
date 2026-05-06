def get_client(self, client_id):
        """
        Returns details about a specific client by the client_id.
        """
        self.assert_has_permission('clients.read')

        uri = self.uri + '/oauth/clients/' + client_id
        headers = self.get_authorization_headers()
        response = requests.get(uri, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            # Not found but don't raise
            return
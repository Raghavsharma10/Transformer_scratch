def list(self):
        """
        Return a list of Accounts from Toshl for the current user
        """
        response = self.client._make_request('/accounts')
        response = response.json()
        return self.client._list_response(response)
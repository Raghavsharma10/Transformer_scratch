def get(self, account_id):
        """
        Return a specific account given its ID
        """
        response = self.client._make_request('/accounts/{0}'.format(account_id))
        return response.json()
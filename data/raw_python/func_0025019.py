def get_scopes(self):
        """
        Returns the scopes for the authenticated client.
        """
        if not self.authenticated:
            raise ValueError("Must authenticate() as a client first.")

        scope = self.client['scope']
        return scope.split()
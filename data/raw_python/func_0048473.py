def search(self, account_name):
        """
        Get a list of all the Accounts for the current user and return the ID
        of the one with the specified name.
        """
        accounts = self.list()
        for a in accounts:
            if a['name'] == account_name:
                return a['id']
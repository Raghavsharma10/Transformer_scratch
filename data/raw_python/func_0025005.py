def create_client(self, client_id, client_secret):
        """
        Create a new client for use by applications.
        """
        assert self.is_admin, "Must authenticate() as admin to create client"
        return self.uaac.create_client(client_id, client_secret)
def change_authentication(self, client_id=None, client_secret=None,
                              access_token=None, refresh_token=None):
        """Change the current authentication."""
        # TODO: Add error checking so you cannot change client_id and retain
        # access_token. Because that doesn't make sense.
        self.client_id = client_id or self.client_id
        self.client_secret = client_secret or self.client_secret
        self.access_token = access_token or self.access_token
        self.refresh_token = refresh_token or self.refresh_token
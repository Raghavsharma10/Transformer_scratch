def get_token(self):
        """
        Returns the bare access token for the authorized client.
        """
        if not self.authenticated:
            raise ValueError("Must authenticate() as a client first.")

        # If token has expired we'll need to refresh and get a new
        # client credential
        if self.is_expired_token(self.client):
            logging.info("client token expired, will need to refresh token")
            self.authenticate(self.client['id'], self.client['secret'],
                    use_cache=False)

        return self.client['access_token']
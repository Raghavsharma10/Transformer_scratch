def valid(self):
        """ Valid credentials are not necessarily correct, but
        they contain all necessary information for an
        authentication attempt. """
        two_legged = self.client_email and self.private_key
        three_legged = self.client_id and self.client_secret
        return two_legged or three_legged or False
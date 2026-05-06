def complete(self):
        """ Complete credentials are valid and are either two-legged or include a token. """
        return self.valid and (self.access_token or self.refresh_token or self.type == 2)
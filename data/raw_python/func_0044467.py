def authenticate(self, token):
        """ Authenticate a token

        :param token:
        """
        if self.verify_token_callback:
            # Specified verify function overrides below
            return self.verify_token_callback(token)

        if not token:
            return False

        name = self.token_manager.verify(token)
        if not name:
            return False

        return True
def authenticate(self):
        """Authenticate user by any means and return either true or false.

        Args:

        Returns:
            tuple (is_valid, username): True is valid user, False if not
        """
        basic_auth = request.authorization
        is_valid = False
        user = None
        if basic_auth:
            is_valid, user = self.check_basic_auth(
                basic_auth.username, basic_auth.password
            )
        else:  # Try token auth
            token = request.headers.get('Authorization', None)
            param_token = request.args.get('access_token')
            if token or param_token:
                if token:
                    # slice the 'token ' piece of the header (following
                    # github style):
                    token = token[6:]
                else:
                    # Grab it from query dict instead
                    token = param_token
                log.debug('Received token: %s', token)

                is_valid, user = self.check_token_auth(token)
        return (is_valid, user)
def token_auth(self, token):
        """Use an application token for authentication.

        :param str token: Application token retrieved from GitHub's
            /authorizations endpoint
        """
        if not token:
            return

        self.headers.update({
            'Authorization': 'token {0}'.format(token)
            })
        # Unset username/password so we stop sending them
        self.auth = None
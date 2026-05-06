def login(self, username=None, password=None, token=None,
              two_factor_callback=None):
        """Logs the user into GitHub for protected API calls.

        :param str username: login name
        :param str password: password for the login
        :param str token: OAuth token
        :param func two_factor_callback: (optional), function you implement to
            provide the Two Factor Authentication code to GitHub when necessary
        """
        if username and password:
            self._session.basic_auth(username, password)
        elif token:
            self._session.token_auth(token)

        # The Session method handles None for free.
        self._session.two_factor_auth_callback(two_factor_callback)
def _login(self, username="", password=""):
        """Login to the Server using username/password,
        empty parameters means an anonymously login
        Returns True if login sucessful, and False if not.
        """
        self.log.debug("----------------")
        self.log.debug("Logging in (username: %s)..." % username)

        def run_query():
            return self._xmlrpc_server.LogIn(
                username, password, self.language, self.user_agent)

        info = self._safe_exec(run_query, None)
        if info is None:
            self._token = None
            return False

        self.log.debug("Login ended in %s with status: %s" %
                       (info['seconds'], info['status']))

        if info['status'] == "200 OK":
            self.log.debug("Session ID: %s" % info['token'])
            self.log.debug("----------------")
            self._token = info['token']
            return True
        else:
            # force token reset
            self.log.debug("----------------")
            self._token = None
            return False
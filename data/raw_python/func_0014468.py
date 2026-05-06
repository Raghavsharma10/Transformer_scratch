def auth_none(self, username):
        """
        Try to authenticate to the server using no authentication at all.
        This will almost always fail.  It may be useful for determining the
        list of authentication types supported by the server, by catching the
        L{BadAuthenticationType} exception raised.

        @param username: the username to authenticate as
        @type username: string
        @return: list of auth types permissible for the next stage of
            authentication (normally empty)
        @rtype: list

        @raise BadAuthenticationType: if "none" authentication isn't allowed
            by the server for this user
        @raise SSHException: if the authentication failed due to a network
            error

        @since: 1.5
        """
        if (not self.active) or (not self.initial_kex_done):
            raise SSHException('No existing session')
        my_event = threading.Event()
        self.auth_handler = AuthHandler(self)
        self.auth_handler.auth_none(username, my_event)
        return self.auth_handler.wait_for_response(my_event)
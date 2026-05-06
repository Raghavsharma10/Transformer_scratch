def start_login_server(self, ):
        """Start a server that will get a request from a user logging in.

        This uses the Implicit Grant Flow of OAuth2. The user is asked
        to login to twitch and grant PyTwitcher authorization.
        Once the user agrees, he is redirected to an url.
        This server will respond to that url and get the oauth token.

        The server serves in another thread. To shut him down, call
        :meth:`TwitchSession.shutdown_login_server`.

        This sets the :data:`TwitchSession.login_server`,
        :data:`TwitchSession.login_thread` variables.

        :returns: The created server
        :rtype: :class:`BaseHTTPServer.HTTPServer`
        :raises: None
        """
        self.login_server = oauth.LoginServer(session=self)
        target = self.login_server.serve_forever
        self.login_thread = threading.Thread(target=target)
        self.login_thread.setDaemon(True)
        log.debug('Starting login server thread.')
        self.login_thread.start()
def connect(self, nick, username=None, realname=None, password=None,
                host=None, port=6667, ssl=None):
        """Connect to the server using the specified credentials.

        Note: if host is specified here, both the host and port arguments
        passed to Client.__init__ will be ignored.

        If the 'ssl' argument is boolean true, will use SSL. If it is a
        dictionary, will both use SSL and pass the contents as kwargs to
        the ssl.wrap_socket() call.
        """
        if host:
            self.server = Host(host, port)
        if self.server is None:
            _log.error("Can't connect() without a host specified.")
            return
        self.user = User(nick)
        self.user.username = username or nick
        self.user.realname = realname or username or nick

        _log.info("Connecting to %s as %s ...", self.server.host, nick)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if ssl and _ssl:
            ssl_kwargs = ssl if isinstance(ssl, dict) else {}
            self.socket = _ssl.wrap_socket(self.socket, **ssl_kwargs)
        elif ssl:
            _log.error("SSL requested but no SSL support available!")
            return

        self.socket.connect((self.server.host, self.server.port))
        self.connected = True

        _log.info("Connected to %s.", self.server.host)

        # Allow an event handler to supply a password instead, if it wants
        suppress_password = self.dispatch_event("PASSWORD")

        if password and not suppress_password:
            # We bypass our own send() function here to avoid logging passwords
            _log.info("Sending server password.")
            self.socket.send("PASS %s\r\n" % password)
            self.server.password = password

        self.dispatch_event('CONNECTED')
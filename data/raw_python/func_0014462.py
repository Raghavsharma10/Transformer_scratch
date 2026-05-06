def start_server(self, event=None, server=None):
        """
        Negotiate a new SSH2 session as a server.  This is the first step after
        creating a new L{Transport} and setting up your server host key(s).  A
        separate thread is created for protocol negotiation.

        If an event is passed in, this method returns immediately.  When
        negotiation is done (successful or not), the given C{Event} will
        be triggered.  On failure, L{is_active} will return C{False}.

        (Since 1.4) If C{event} is C{None}, this method will not return until
        negotation is done.  On success, the method returns normally.
        Otherwise an SSHException is raised.

        After a successful negotiation, the client will need to authenticate.
        Override the methods
        L{get_allowed_auths <ServerInterface.get_allowed_auths>},
        L{check_auth_none <ServerInterface.check_auth_none>},
        L{check_auth_password <ServerInterface.check_auth_password>}, and
        L{check_auth_publickey <ServerInterface.check_auth_publickey>} in the
        given C{server} object to control the authentication process.

        After a successful authentication, the client should request to open
        a channel.  Override
        L{check_channel_request <ServerInterface.check_channel_request>} in the
        given C{server} object to allow channels to be opened.

        @note: After calling this method (or L{start_client} or L{connect}),
            you should no longer directly read from or write to the original
            socket object.

        @param event: an event to trigger when negotiation is complete.
        @type event: threading.Event
        @param server: an object used to perform authentication and create
            L{Channel}s.
        @type server: L{server.ServerInterface}

        @raise SSHException: if negotiation fails (and no C{event} was passed
            in)
        """
        if server is None:
            server = ServerInterface()
        self.server_mode = True
        self.server_object = server
        self.active = True
        if event is not None:
            # async, return immediately and let the app poll for completion
            self.completion_event = event
            self.start()
            return

        # synchronous, wait for a result
        self.completion_event = event = threading.Event()
        self.start()
        while True:
            event.wait(0.1)
            if not self.active:
                e = self.get_exception()
                if e is not None:
                    raise e
                raise SSHException('Negotiation failed.')
            if event.isSet():
                break
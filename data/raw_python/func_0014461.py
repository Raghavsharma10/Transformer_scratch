def start_client(self, event=None):
        """
        Negotiate a new SSH2 session as a client.  This is the first step after
        creating a new L{Transport}.  A separate thread is created for protocol
        negotiation.

        If an event is passed in, this method returns immediately.  When
        negotiation is done (successful or not), the given C{Event} will
        be triggered.  On failure, L{is_active} will return C{False}.

        (Since 1.4) If C{event} is C{None}, this method will not return until
        negotation is done.  On success, the method returns normally.
        Otherwise an SSHException is raised.

        After a successful negotiation, you will usually want to authenticate,
        calling L{auth_password <Transport.auth_password>} or
        L{auth_publickey <Transport.auth_publickey>}.

        @note: L{connect} is a simpler method for connecting as a client.

        @note: After calling this method (or L{start_server} or L{connect}),
            you should no longer directly read from or write to the original
            socket object.

        @param event: an event to trigger when negotiation is complete
            (optional)
        @type event: threading.Event

        @raise SSHException: if negotiation fails (and no C{event} was passed
            in)
        """
        self.active = True
        if event is not None:
            # async, return immediately and let the app poll for completion
            self.completion_event = event
            self.start()
            return

        # synchronous, wait for a result
        self.completion_event = event = threading.Event()
        self.start()
        Random.atfork()
        while True:
            event.wait(0.1)
            if not self.active:
                e = self.get_exception()
                if e is not None:
                    raise e
                raise SSHException('Negotiation failed.')
            if event.isSet():
                break
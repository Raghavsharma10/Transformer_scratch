def authenticationrequest(self, event):
        """Handles authentication requests from clients
        :param event: AuthenticationRequest with user's credentials
        """

        if event.sock.getpeername()[0] in self.failing_clients:
            self.log('Client failed a login and has to wait', lvl=debug)
            return

        if event.auto:
            self._handle_autologin(event)
        else:
            self._handle_login(event)
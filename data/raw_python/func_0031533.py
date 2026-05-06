def connect(self, user=None):
        """ Connect to host.

        :param user: if user - login session.
        """

        self.api._tcl_handler.connect()
        if user:
            self.session.login(user)
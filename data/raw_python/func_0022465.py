def check_session(self):
        """
        Make sure a session is open.

        If it's not and autosession is turned on, create a new session automatically.
        If it's not and autosession is off, raise an exception.
        """
        if self.session is None:
            if self.autosession:
                self.open_session()
            else:
                msg = "must open a session before modifying %s" % self
                raise RuntimeError(msg)
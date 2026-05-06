def keep_session_alive(self):
        """If the session expired, logs back in."""

        try:
            self.resources()
        except xmlrpclib.Fault as fault:
            if fault.faultCode == 5:
                self.login()
            else:
                raise
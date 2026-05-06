def handle(self, error, connection):
        """ Handle any cleanup or similar activity related to an error
        occurring on a pooled connection.
        """
        error_class = error.__class__
        if error_class in (ConnectionExpired, ServiceUnavailable, DatabaseUnavailableError):
            self.deactivate(connection.address)
        elif error_class in (NotALeaderError, ForbiddenOnReadOnlyDatabaseError):
            self.remove_writer(connection.address)
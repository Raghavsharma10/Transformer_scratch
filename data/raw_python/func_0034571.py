def verbose(self, msg, *args, **kw):
        """Log a message with level :data:`VERBOSE`. The arguments are interpreted as for :func:`logging.debug()`."""
        if self.isEnabledFor(VERBOSE):
            self._log(VERBOSE, msg, args, **kw)
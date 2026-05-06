def success(self, msg, *args, **kw):
        """Log a message with level :data:`SUCCESS`. The arguments are interpreted as for :func:`logging.debug()`."""
        if self.isEnabledFor(SUCCESS):
            self._log(SUCCESS, msg, args, **kw)
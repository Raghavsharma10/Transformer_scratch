def spam(self, msg, *args, **kw):
        """Log a message with level :data:`SPAM`. The arguments are interpreted as for :func:`logging.debug()`."""
        if self.isEnabledFor(SPAM):
            self._log(SPAM, msg, args, **kw)
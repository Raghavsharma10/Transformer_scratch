def notice(self, msg, *args, **kw):
        """Log a message with level :data:`NOTICE`. The arguments are interpreted as for :func:`logging.debug()`."""
        if self.isEnabledFor(NOTICE):
            self._log(NOTICE, msg, args, **kw)
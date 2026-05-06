def _got_request_exception(self, sender, exception, **extra):
        """
        The signal handler for the got_request_exception signal.
        """
        extra = self.summary_extra()
        extra['errno'] = 500
        self.summary_logger.error(str(exception), extra=extra)
        g._has_exception = True
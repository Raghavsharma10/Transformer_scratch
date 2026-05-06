def log_error(cls, msg):
        """
        Logs the provided error message to both the error logger and the debug logger logging
        instances.

        Args:
            msg: `str`. The error message to log.
        """
        cls.error_logger.error(msg)
        cls.debug_logger.debug(msg)
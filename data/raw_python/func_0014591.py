def log(self, level, msg):
        """Write a diagnostic message to a log file or to standard output.

        Arguments:
        level -- Severity level of entry.  One of: INFO, WARN, ERROR, FATAL.
        msg   -- Message to write to log.

        """
        self._check_session()
        level = level.upper()
        allowed_levels = ('INFO', 'WARN', 'ERROR', 'FATAL')
        if level not in allowed_levels:
            raise ValueError('level must be one of: ' +
                             ', '.join(allowed_levels))
        self._rest.post_request(
            'log', None, {'log_level': level.upper(), 'message': msg})
def set_log_level(cls, log_level):
        """Sets the log level for cons3rt assets

        This method sets the logging level for cons3rt assets using
        pycons3rt. The loglevel is read in from a deployment property
        called loglevel and set appropriately.

        :type log_level: str
        :return: True if log level was set, False otherwise.
        """
        log = logging.getLogger(cls.cls_logger + '.set_log_level')
        log.info('Attempting to set the log level...')
        if log_level is None:
            log.info('Arg loglevel was None, log level will not be updated.')
            return False
        if not isinstance(log_level, basestring):
            log.error('Passed arg loglevel must be a string')
            return False
        log_level = log_level.upper()
        log.info('Attempting to set log level to: %s...', log_level)
        if log_level == 'DEBUG':
            cls._logger.setLevel(logging.DEBUG)
        elif log_level == 'INFO':
            cls._logger.setLevel(logging.INFO)
        elif log_level == 'WARN':
            cls._logger.setLevel(logging.WARN)
        elif log_level == 'WARNING':
            cls._logger.setLevel(logging.WARN)
        elif log_level == 'ERROR':
            cls._logger.setLevel(logging.ERROR)
        else:
            log.error('Could not set log level, this is not a valid log level: %s', log_level)
            return False
        log.info('pycons3rt loglevel set to: %s', log_level)
        return True
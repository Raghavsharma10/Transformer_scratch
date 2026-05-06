def configureLogger(self):
        """
        Configures the python logging system to log to a debug file and to stdout for warn and above.
        :return: the base logger.
        """
        baseLogLevel = logging.DEBUG if self.isDebugLogging() else logging.INFO
        # create recorder app root logger
        logger = logging.getLogger(self._name)
        logger.setLevel(baseLogLevel)
        # file handler
        fh = handlers.RotatingFileHandler(path.join(self._getConfigPath(), self._name + '.log'),
                                          maxBytes=10 * 1024 * 1024, backupCount=10)
        fh.setLevel(baseLogLevel)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger
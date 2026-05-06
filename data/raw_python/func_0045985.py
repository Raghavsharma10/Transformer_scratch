def set_handler(self, log, handler):
        """Set the logging level as well as the handlers.

        :param log: ``object``
        :param handler: ``object``
        """
        if self.debug_logging is True:
            log.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)
            handler.setLevel(logging.INFO)

        handler.name = self.name
        handler.setFormatter(self.format)
        log.addHandler(handler)
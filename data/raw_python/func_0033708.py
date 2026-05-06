def set_log_level(self, level: str) -> None:
        """Override the default log level of the class."""
        if level == 'info':
            to_set = logging.INFO
        if level == 'debug':
            to_set = logging.DEBUG
        if level == 'error':
            to_set = logging.ERROR
        self.log.setLevel(to_set)
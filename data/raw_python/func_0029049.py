def configure(self):
        """Configure the Python stdlib logger"""
        if self.debug is not None and not self.debug:
            self._remove_debug_handlers()
        self._remove_debug_only()
        logging.config.dictConfig(self.config)
        try:
            logging.captureWarnings(True)
        except AttributeError:
            pass
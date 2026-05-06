def recover_all_handler(self):
        """
        Relink the file handler association you just removed.
        """
        for handler in self._handler_cache:
            self.logger.addHandler(handler)
        self._handler_cache = list()
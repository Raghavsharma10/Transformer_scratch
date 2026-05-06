def remove_all_handler(self):
        """
        Unlink the file handler association.
        """
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            self._handler_cache.append(handler)
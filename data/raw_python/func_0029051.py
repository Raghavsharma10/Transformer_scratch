def _remove_debug_only(self):
        """Iterate through each handler removing the invalid dictConfig key of
        debug_only.

        """
        LOGGER.debug('Removing debug only from handlers')
        for handler in self.config[self.HANDLERS]:
            if self.DEBUG_ONLY in self.config[self.HANDLERS][handler]:
                del self.config[self.HANDLERS][handler][self.DEBUG_ONLY]
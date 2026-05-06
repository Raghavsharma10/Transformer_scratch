def _remove_debug_handlers(self):
        """Remove any handlers with an attribute of debug_only that is True and
        remove the references to said handlers from any loggers that are
        referencing them.

        """
        remove = list()
        for handler in self.config[self.HANDLERS]:
            if self.config[self.HANDLERS][handler].get('debug_only'):
                remove.append(handler)
        for handler in remove:
            del self.config[self.HANDLERS][handler]
            for logger in self.config[self.LOGGERS].keys():
                logger = self.config[self.LOGGERS][logger]
                if handler in logger[self.HANDLERS]:
                    logger[self.HANDLERS].remove(handler)
        self._remove_debug_only()
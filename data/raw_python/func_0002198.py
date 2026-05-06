def register(self, mimetype):
        """Register a function to handle a particular mimetype."""
        def dec(func):
            self._reg[mimetype] = func
            return func
        return dec
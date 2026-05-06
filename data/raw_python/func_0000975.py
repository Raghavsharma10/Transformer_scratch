def handler(self, handler_class):
        """Link to an API handler class (e.g. piston or DRF)."""
        self.handler_class = handler_class
        # we take the docstring from the handler class, not the methods
        if self.docs is None and handler_class.__doc__:
            self.docs = clean_docstring(handler_class.__doc__)
        return handler_class
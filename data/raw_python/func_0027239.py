def subscribe(self, handler):
        """Adds a new event handler."""
        assert callable(handler), "Invalid handler %s" % handler
        self.handlers.append(handler)
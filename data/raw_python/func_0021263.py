def on_ignored(self, metadata=None):
        """ Called when an IGNORED message has been received.
        """
        handler = self.handlers.get("on_ignored")
        if callable(handler):
            handler(metadata)
        handler = self.handlers.get("on_summary")
        if callable(handler):
            handler()
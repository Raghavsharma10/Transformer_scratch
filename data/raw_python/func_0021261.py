def on_success(self, metadata):
        """ Called when a SUCCESS message has been received.
        """
        handler = self.handlers.get("on_success")
        if callable(handler):
            handler(metadata)
        handler = self.handlers.get("on_summary")
        if callable(handler):
            handler()
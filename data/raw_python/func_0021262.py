def on_failure(self, metadata):
        """ Called when a FAILURE message has been received.
        """
        self.connection.reset()
        handler = self.handlers.get("on_failure")
        if callable(handler):
            handler(metadata)
        handler = self.handlers.get("on_summary")
        if callable(handler):
            handler()
        raise CypherError.hydrate(**metadata)
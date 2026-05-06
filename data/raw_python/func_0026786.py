def _dispatcher(self, connection, event):
        """
        This is the method in ``SimpleIRCClient`` that all IRC events
        get passed through. Here we map events to our own custom
        event handlers, and call them.
        """
        super(BaseBot, self)._dispatcher(connection, event)
        for handler in self.events[event.eventtype()]:
            handler(self, connection, event)
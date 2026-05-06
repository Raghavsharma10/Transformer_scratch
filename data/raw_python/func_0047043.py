def add_handler(self, event, handler):
        """Adds a handler function for an event.

        Note: Only one handler function is allowed per event on the module
        level (different modules can provide handlers for the same event).
        This is because ordering of handler functions is not guaranteed to
        be preserved at the module level.

        Also note that it's probably easier and more succint to use the
        decorator form of this e.g. @Module.handle('EVENT')
        """
        if event in self.event_handlers:
            raise ValueError("Cannot register handler for '%s' twice." % event)
        self.event_handlers[event] = handler
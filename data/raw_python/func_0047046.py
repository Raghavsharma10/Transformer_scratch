def handle_event(self, event, client, args):
        """Dispatch an event to its handler.

        Note: the handler does not receive the event which triggered its call.
        If you want to handle more than one event, it's recommended to put the
        shared handling in a separate function, and create wrapper handlers
        that call the shared function.
        """
        handler = self.event_handlers.get(event)
        if handler:
            return handler(client, *args)
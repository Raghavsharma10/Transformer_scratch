def listen(self, event):
        """Request that the Controller listen for and dispatch an event.

        Note: Even if the module that requested the listening is later
        unloaded, the Controller will continue to dispatch the event, there
        just might not be anything that cares about it. That's okay.
        """
        if event in self.registered:
            # Already listening to this event
            return
        def handler(client, *args):
            return self.process_event(event, client, args)
        self.client.add_handler(event, handler)
        self.registered.add(event)
        _log.debug("Controller is now listening for '%s' events", event)
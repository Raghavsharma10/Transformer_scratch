def dispatch_event(self, event, *args):
        """Dispatches an event.

        Returns a boolean indicating whether or not a handler
        suppressed further handling of the event (even the last).
        """
        if event not in self.event_handlers:
            _log.error("Dispatch requested for unknown event '%s'", event)
            return False
        elif event != "LINE":
            _log.debug("Dispatching event %s %r", event, args)

        try:
            for handler in self.event_handlers[event]:
                # (client, server, *args) : args are dependent on event
                if handler(self, *args):
                    # Returning a truthy value supresses further handlers
                    # for this event.
                    return True
        except Exception as e:
            _log.exception("Error while processing event '%s': %r", event, e)

        # Fall back to the RAWLINE event if LINE can't process it.
        if event == "LINE":
            return self.dispatch_event("RAWLINE", *args)

        return False
def add_handler(self, event, handler):
        """Adds a handler for a particular event.

        Handlers are appended to the list, so a handler added earlier
        will be called before a handler added later. If you wish to
        insert a handler at another position, you should modify the
        event_handlers property directly:

            my_client.event_handlers['PRIVMSG'].insert(0, my_handler)
        """
        if event not in self.event_handlers:
            _log.info("Adding event handler for new event %s.", event)
            self.event_handlers[event] = [handler]
        else:
            self.event_handlers[event].append(handler)
def off(self, event, handler):
        """Detaches the handler from the specified event.

        @param event: event to detach the handler to. Any object can be passed
                      as event, but string is preferable. If qcore.EnumBase
                      instance is passed, its name is used as event key.
        @param handler: event handler.
        @return: self, so calls like this can be chained together.

        """
        event_hook = self.get_or_create(event)
        event_hook.unsubscribe(handler)
        return self
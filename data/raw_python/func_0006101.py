def bind(self, event_name, callback, *args, **kwargs):
        """Bind an event to a callback

        :param event_name: The name of the event to bind to.
        :type event_name: str

        :param callback: The callback to notify of this event.
        """
        self.event_callbacks[event_name].append((callback, args, kwargs))
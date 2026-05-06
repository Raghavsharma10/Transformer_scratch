def filter_events(cls, client, event_data):
        """Filter registered events and yield them."""

        for event in cls.events:
            # try event filters
            if event.matches(client, event_data):
                yield event
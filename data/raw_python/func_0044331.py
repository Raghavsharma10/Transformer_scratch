def filter_callbacks(cls, client, event_data):
        """Filter registered events and yield all of their callbacks."""

        for event in cls.filter_events(client, event_data):
            for cb in event.callbacks:
                yield cb
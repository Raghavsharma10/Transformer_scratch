def scheduled_event_trigger(self, event_type):
        """Returns a callback that schedules events for the future.

        Returned callback function will add an event of type event_type
        to a queue which will be checked the next time an event is requested."""
        def callback(when, **kwargs):
            self.queued_scheduled_events.append((when, event_type(when=when, **kwargs)))
        return callback
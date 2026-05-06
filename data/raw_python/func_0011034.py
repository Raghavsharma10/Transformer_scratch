def event_trigger(self, event_type):
        """Returns a callback that creates events.

        Returned callback function will add an event of type event_type
        to a queue which will be checked the next time an event is requested."""
        def callback(**kwargs):
            self.queued_events.append(event_type(**kwargs))
        return callback
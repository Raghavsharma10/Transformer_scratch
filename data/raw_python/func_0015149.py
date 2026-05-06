def pop_event(self):
        """Pop an event from event_list."""
        if len(self.event_list) > 0:
            evt = self.event_list.pop(0)
            return evt
        return None
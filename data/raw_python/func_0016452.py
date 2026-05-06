def drain_events(self, allowed_methods=None, timeout=None):
        """Wait for an event on any channel."""
        return self.wait_multi(self.channels.values(), timeout=timeout)
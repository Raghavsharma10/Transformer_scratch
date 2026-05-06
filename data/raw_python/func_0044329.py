def matches(self, client, event_data):
        """True if all filters are matching."""

        for f in self.filters:
            if not f(client, event_data):
                return False

        return True
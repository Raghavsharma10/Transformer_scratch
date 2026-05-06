def time_between_updates(self):
        """Time between current `last_updated` and previous `last_updated`"""
        if 'last_updated' not in self._original:
            return 0
        last_update = self._original['last_updated']
        this_update = self.last_updated
        return this_update - last_update
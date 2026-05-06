def is_open(self, refresh=False):
        """Get curtains state.

        Refresh data from Vera if refresh is True, otherwise use local cache.
        Refresh is only needed if you're not using subscriptions.
        """
        if refresh:
            self.refresh()
        return self.get_level(refresh) > 0
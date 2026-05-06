def is_switched_on(self, refresh=False):
        """Get armed state.

        Refresh data from Vera if refresh is True, otherwise use local cache.
        Refresh is only needed if you're not using subscriptions.
        """
        if refresh:
            self.refresh()
        val = self.get_value('Armed')
        return val == '1'
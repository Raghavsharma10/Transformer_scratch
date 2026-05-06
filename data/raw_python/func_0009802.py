def is_switched_on(self, refresh=False):
        """Get dimmer state.

        Refresh data from Vera if refresh is True,
        otherwise use local cache. Refresh is only needed if you're
        not using subscriptions.
        """
        if refresh:
            self.refresh()
        return self.get_brightness(refresh) > 0
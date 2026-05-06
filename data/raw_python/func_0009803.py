def get_brightness(self, refresh=False):
        """Get dimmer brightness.

        Refresh data from Vera if refresh is True, otherwise use local cache.
        Refresh is only needed if you're not using subscriptions.
        Converts the Vera level property for dimmable lights from a percentage
        to the 0 - 255 scale used by HA.
        """
        if refresh:
            self.refresh()
        brightness = 0
        percent = self.level
        if percent > 0:
            brightness = round(percent * 2.55)
        return int(brightness)
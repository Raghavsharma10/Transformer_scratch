def get_last_scene_time(self, refresh=False):
        """Get last scene time.

        Refresh data from Vera if refresh is True, otherwise use local cache.
        Refresh is only needed if you're not using subscriptions.
        """
        if refresh:
            self.refresh_complex_value('LastSceneTime')
        val = self.get_complex_value('LastSceneTime')
        return val
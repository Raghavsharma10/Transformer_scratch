def get_last_scene_id(self, refresh=False):
        """Get last scene id.

        Refresh data from Vera if refresh is True, otherwise use local cache.
        Refresh is only needed if you're not using subscriptions.
        """
        if refresh:
            self.refresh_complex_value('LastSceneID')
            self.refresh_complex_value('sl_CentralScene')
        val = self.get_complex_value('LastSceneID') or self.get_complex_value('sl_CentralScene')
        return val
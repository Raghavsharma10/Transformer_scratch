def monitor(self):
        """Property providing access to the :class:`.MonitorAPI`"""
        if self._monitor_api is None:
            self._monitor_api = self.get_monitor_api()
        return self._monitor_api
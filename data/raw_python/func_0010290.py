def devicecore(self):
        """Property providing access to the :class:`.DeviceCoreAPI`"""
        if self._devicecore_api is None:
            self._devicecore_api = self.get_devicecore_api()
        return self._devicecore_api
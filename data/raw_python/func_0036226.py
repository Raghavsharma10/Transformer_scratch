def register_callback(self, sensorid, callback, user_data=None):
        """Register a callback for the specified sensor id."""
        if sensorid not in self._registry:
            self._registry[sensorid] = list()
        self._registry[sensorid].append((callback, user_data))
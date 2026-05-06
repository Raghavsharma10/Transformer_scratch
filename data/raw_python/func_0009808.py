def set_armed_state(self, state):
        """Set the armed state, also update local state."""
        self.set_service_value(
            self.security_sensor_service,
            'Armed',
            'newArmedValue',
            state)
        self.set_cache_value('Armed', state)
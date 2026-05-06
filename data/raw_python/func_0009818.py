def set_fan_mode(self, mode):
        """Set the fan mode"""
        self.set_service_value(
            self.thermostat_fan_service,
            'Mode',
            'NewMode',
            mode)
        self.set_cache_value('fanmode', mode)
def set_hvac_mode(self, mode):
        """Set the hvac mode"""
        self.set_service_value(
            self.thermostat_operating_service,
            'ModeTarget',
            'NewModeTarget',
            mode)
        self.set_cache_value('mode', mode)
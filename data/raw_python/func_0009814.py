def set_temperature(self, temp):
        """Set current goal temperature / setpoint"""

        self.set_service_value(
            self.thermostat_setpoint,
            'CurrentSetpoint',
            'NewCurrentSetpoint',
            temp)

        self.set_cache_value('setpoint', temp)
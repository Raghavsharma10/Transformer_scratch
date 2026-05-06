def device_status(self):
        """Status of device."""
        return {
            'active': self.device['active'],
            'offline': self.device['offline'],
            'last_update': self.last_update,
            'battery_level': self.battery_level,
        }
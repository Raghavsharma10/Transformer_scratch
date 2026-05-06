def device_info(self):
        """Info about device."""
        return {
            'connections': {('mac', self.device['device_mac'])},
            'identifieres': self.device['device_id'],
            'manufacturer': 'Minut',
            'model': 'Point v{}'.format(self.device['hardware_version']),
            'name': self.device['description'],
            'sw_version': self.device['firmware']['installed'],
        }
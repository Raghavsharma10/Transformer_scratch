def get_mac(self, use_cached=True):
        """Get the MAC address of this device"""
        device_json = self.get_device_json(use_cached)
        return device_json.get("devMac")
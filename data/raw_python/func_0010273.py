def get_ip(self, use_cached=True):
        """Get the last known IP of this device"""
        device_json = self.get_device_json(use_cached)
        return device_json.get("dpLastKnownIp")
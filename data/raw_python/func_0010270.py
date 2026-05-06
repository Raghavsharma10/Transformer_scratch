def is_connected(self, use_cached=True):
        """Return True if the device is currrently connect and False if not"""
        device_json = self.get_device_json(use_cached)
        return int(device_json.get("dpConnectionStatus")) > 0
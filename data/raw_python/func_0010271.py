def get_connectware_id(self, use_cached=True):
        """Get the connectware id of this device (primary key)"""
        device_json = self.get_device_json(use_cached)
        return device_json.get("devConnectwareId")
def get_device_id(self, use_cached=True):
        """Get this device's device id"""
        device_json = self.get_device_json(use_cached)
        return device_json["id"].get("devId")
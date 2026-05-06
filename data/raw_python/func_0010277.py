def get_latlon(self, use_cached=True):
        """Get a tuple with device latitude and longitude... these may be None"""
        device_json = self.get_device_json(use_cached)
        lat = device_json.get("dpMapLat")
        lon = device_json.get("dpMapLong")
        return (float(lat) if lat else None,
                float(lon) if lon else None, )
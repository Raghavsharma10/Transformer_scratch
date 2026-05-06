def get_tags(self, use_cached=True):
        """Get the list of tags for this device"""
        device_json = self.get_device_json(use_cached)
        potential_tags = device_json.get("dpTags")
        if potential_tags:
            return list(filter(None, potential_tags.split(",")))
        else:
            return []
def set_property(self, key, value):
        """
        Update only one property in the dict
        """
        self.properties[key] = value
        self.sync_properties()
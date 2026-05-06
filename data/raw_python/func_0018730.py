def set_properties(self, properties):
        """Sets properties and text of this info from a dictionary"""
        if isinstance(properties, dict):
            self.properties = properties
            self.sync_properties()
        else:
            self.text = properties
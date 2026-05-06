def set_description(self, description):
        """Set the description for this data point"""
        self._description = validate_type(description, type(None), *six.string_types)
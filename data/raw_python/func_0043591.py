def use_value(self, value):
        """Converts value to field type or use original"""
        if self.check_value(value):
            return value
        return self.convert_value(value)
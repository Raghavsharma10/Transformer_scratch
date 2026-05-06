def parse_value(self, values):
        """Cast value to proper collection."""
        result = self.get_default_value()

        if not values:
            return result

        if not isinstance(values, list):
            return values

        return [self._cast_value(value) for value in values]
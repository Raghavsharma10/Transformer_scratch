def clear_list_value(self, value):
        """
        Clean the argument value to eliminate None or Falsy values if needed.
        """
        # Don't go any further: this value is empty.
        if not value:
            return self.empty_value
        # Clean empty items if wanted
        if self.clean_empty:
            value = [v for v in value if v]
        return value or self.empty_value
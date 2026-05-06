def validate(self, value):
        """
        Applies the validation criteria.
        Returns value, new value, or None if invalid.

        Overload this in derived classes.
        """
        try:
            v = int(value)
            if v not in self.allowed:
                return None
            return value
        except ValueError:
            return None
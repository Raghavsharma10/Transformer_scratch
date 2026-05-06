def validate(self, value):
        """
        Applies the validation criteria.
        Returns value, new value, or None if invalid.
        """
        try:
            coord.Angle(value, unit=self.unit)
            return value
        except ValueError:
            return None
def validate(self, value):
        """
        Applies the validation criteria.
        Returns value, new value, or None if invalid.

        Overload this in derived classes.
        """
        try:
            # trap blank fields here
            if not self.blank or value:
                v = float(value)
                if (self.allowzero and v != 0 and v < self.fmin) or \
                        (not self.allowzero and v < self.fmin) or v > self.fmax:
                    return None
            return value
        except ValueError:
            return None
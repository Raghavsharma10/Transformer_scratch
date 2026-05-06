def validate(self, value):
        """
        This prevents setting any value more precise than 0.00001
        """
        try:
            # trap blank fields here
            if value:
                v = float(value)
                if (v != 0 and v < self.fmin) or v > self.fmax:
                    return None
                if abs(round(100000*v)-100000*v) > 1.e-12:
                    return None
            return value
        except ValueError:
            return None
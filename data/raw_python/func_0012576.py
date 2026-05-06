def get_checks(self):
        """Return a list of functions to use when testing values."""
        return [
            self.is_date,
            self.is_datetime,
            self.is_integer,
            self.is_float,
            self.default]
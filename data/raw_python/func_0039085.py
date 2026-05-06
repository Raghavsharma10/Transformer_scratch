def safe_round(self, x):
        """Returns a converter that takes in a value and turns it into an integer, if necessary.

        Args:
            col_name(str): Name of the column.
            subtype(str): Numeric subtype of the values.

        Returns:
            function
        """
        val = x[self.col_name]

        if np.isposinf(val):
            val = sys.maxsize

        elif np.isneginf(val):
            val = -sys.maxsize

        if np.isnan(val):
            val = self.default_val

        if self.subtype == 'integer':
            return int(round(val))

        return val
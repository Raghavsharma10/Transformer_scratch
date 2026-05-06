def _check_constant_value_data(self, data):
        """Verify that the HDU's data is a constant value array."""

        arrayval = data.flat[0]
        if np.all(data == arrayval):
            return arrayval
        return None
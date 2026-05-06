def _apply_data(self, f, ts, reverse=False):
        """
        Convenience function for all of the math stuff.
        """
        # TODO: needs to catch np numeric types?
        if isinstance(ts, (int, float)):
            d = ts * np.ones(self.shape[0])
        elif ts is None:
            d = None
        elif np.array_equal(ts.index, self.index):
            d = ts.values
        else:
            d = ts._retime(self.index)

        if not reverse:
            new_data = np.apply_along_axis(f, 0, self.values, d)
        else:
            new_data = np.apply_along_axis(f, 0, d, self.values)
        return Trace(new_data, self.index, name=self.name)
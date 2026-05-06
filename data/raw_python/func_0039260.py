def row_value(self, row):
        """Get the tracks value at row"""
        irow = int(row)
        i = self._get_key_index(irow)
        if i == -1:
            return 0.0

        # Are we dealing with the last key?
        if i == len(self.keys) - 1:
            return self.keys[-1].value

        return TrackKey.interpolate(self.keys[i], self.keys[i + 1], row)
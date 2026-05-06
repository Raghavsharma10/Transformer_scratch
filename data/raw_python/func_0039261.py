def add_or_update(self, row, value, kind):
        """Add or update a track value"""
        i = bisect.bisect_left(self.keys, row)

        # Are we simply replacing a key?
        if i < len(self.keys) and self.keys[i].row == row:
            self.keys[i].update(value, kind)
        else:
            self.keys.insert(i, TrackKey(row, value, kind))
def _get_key_index(self, row):
        """Get the key that should be used as the first interpolation value"""
        # Don't bother with empty tracks
        if len(self.keys) == 0:
            return -1

        # No track values are defined yet
        if row < self.keys[0].row:
            return -1

        # Get the insertion index
        index = bisect.bisect_left(self.keys, row)
        # Index is within the array size?
        if index < len(self.keys):
            # Are we inside an interval?
            if row < self.keys[index].row:
                return index - 1
            return index

        # Return the last index
        return len(self.keys) - 1
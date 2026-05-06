def _lookup(self, timestamp):
        """
        Return the index of the value associated with "timestamp" if any, else
        None. Since the timestamps are floating-point values, they are
        considered equal if their absolute difference is smaller than
        self.EPSILON
        """

        idx = search_greater(self._values, timestamp)
        if (idx < len(self._values)
                and math.fabs(self._values[idx][0] - timestamp) < self.EPSILON):
            return idx
        return None
def record_value(self, value, count=1):
        '''Record a new value into the histogram

        Args:
            value: the value to record (must be in the valid range)
            count: incremental count (defaults to 1)
        '''
        if value < 0:
            return False
        counts_index = self._counts_index_for(value)
        if (counts_index < 0) or (self.counts_len <= counts_index):
            return False
        self.counts[counts_index] += count
        self.total_count += count
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        return True
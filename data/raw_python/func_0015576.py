def get_value_at_percentile(self, percentile):
        '''Get the value for a given percentile

        Args:
            percentile: a float in [0.0..100.0]
        Returns:
            the value for the given percentile
        '''
        count_at_percentile = self.get_target_count_at_percentile(percentile)
        total = 0
        for index in range(self.counts_len):
            total += self.get_count_at_index(index)
            if total >= count_at_percentile:
                value_at_index = self.get_value_from_index(index)
                if percentile:
                    return self.get_highest_equivalent_value(value_at_index)
                return self.get_lowest_equivalent_value(value_at_index)
        return 0
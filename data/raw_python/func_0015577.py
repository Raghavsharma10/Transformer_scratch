def get_percentile_to_value_dict(self, percentile_list):
        '''A faster alternative to query values for a list of percentiles.

        Args:
            percentile_list: a list of percentiles in any order, dups will be ignored
            each element in the list must be a float value in [0.0 .. 100.0]
        Returns:
            a dict of percentile values indexed by the percentile
        '''
        result = {}
        total = 0
        percentile_list_index = 0
        count_at_percentile = 0
        # remove dups and sort
        percentile_list = list(set(percentile_list))
        percentile_list.sort()

        for index in range(self.counts_len):
            total += self.get_count_at_index(index)
            while True:
                # recalculate target based on next requested percentile
                if not count_at_percentile:
                    if percentile_list_index == len(percentile_list):
                        return result
                    percentile = percentile_list[percentile_list_index]
                    percentile_list_index += 1
                    if percentile > 100:
                        return result
                    count_at_percentile = self.get_target_count_at_percentile(percentile)

                if total >= count_at_percentile:
                    value_at_index = self.get_value_from_index(index)
                    if percentile:
                        result[percentile] = self.get_highest_equivalent_value(value_at_index)
                    else:
                        result[percentile] = self.get_lowest_equivalent_value(value_at_index)
                    count_at_percentile = 0
                else:
                    break
        return result
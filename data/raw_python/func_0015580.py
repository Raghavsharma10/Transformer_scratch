def adjust_internal_tacking_values(self,
                                       min_non_zero_index,
                                       max_index,
                                       total_added):
        '''Called during decoding and add to adjust the new min/max value and
        total count

        Args:
            min_non_zero_index min nonzero index of all added counts (-1 if none)
            max_index max index of all added counts (-1 if none)
        '''
        if max_index >= 0:
            max_value = self.get_highest_equivalent_value(self.get_value_from_index(max_index))
            self.max_value = max(self.max_value, max_value)
        if min_non_zero_index >= 0:
            min_value = self.get_value_from_index(min_non_zero_index)
            self.min_value = min(self.min_value, min_value)
        self.total_count += total_added
def set_internal_tacking_values(self,
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
            self.max_value = self.get_highest_equivalent_value(self.get_value_from_index(max_index))
        if min_non_zero_index >= 0:
            self.min_value = self.get_value_from_index(min_non_zero_index)
        self.total_count = total_added
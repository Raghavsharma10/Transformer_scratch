def values_are_equivalent(self, val1, val2):
        '''Check whether 2 values are equivalent (meaning they
        are in the same bucket/range)

        Returns:
            true if the 2 values are equivalent
        '''
        return self.get_lowest_equivalent_value(val1) == self.get_lowest_equivalent_value(val2)
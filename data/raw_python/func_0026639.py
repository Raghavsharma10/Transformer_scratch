def clipValue(self, value, minValue, maxValue):
        '''
        Makes sure that value is within a specific range.
        If not, then the lower or upper bounds is returned
        '''
        return min(max(value, minValue), maxValue)
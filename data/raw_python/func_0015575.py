def record_corrected_value(self, value, expected_interval, count=1):
        '''Record a new value into the histogram and correct for
        coordinated omission if needed

        Args:
            value: the value to record (must be in the valid range)
            expected_interval: the expected interval between 2 value samples
            count: incremental count (defaults to 1)
        '''
        while True:
            if not self.record_value(value, count):
                return False
            if value <= expected_interval or expected_interval <= 0:
                return True
            value -= expected_interval
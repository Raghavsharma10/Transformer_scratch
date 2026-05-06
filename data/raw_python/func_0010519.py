def set_range(self, minimum, maximum):
        """
        Set a range.
        The range is passed unchanged to the rangeChanged member function.
        :param minimum: minimum value of the range (None if no percentage is required)
        :param maximum: maximum value of the range (None if no percentage is required)
        """
        self._min = minimum
        self._max = maximum
        self.on_rangeChange(minimum, maximum)
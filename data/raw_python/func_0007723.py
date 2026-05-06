def set_value(self, value: str):
        """
        Sets the displayed digits based on the value string.
        :param value: a string containing an integer or float value
        :return: None
        """
        [digit.clear() for digit in self._digits]

        grouped = self._group(value)  # return the parts, reversed
        digits = self._digits[::-1]  # reverse the digits

        # fill from right to left
        has_period = False
        for i, digit_value in enumerate(grouped):
            try:
                if has_period:
                    digits[i].set_value(digit_value + '.')
                    has_period = False

                elif grouped[i] == '.':
                    has_period = True

                else:
                    digits[i].set_value(digit_value)
            except IndexError:
                raise ValueError('the value "{}" contains too '
                                 'many digits'.format(value))
def set(self, value: int):
        """
        Set the current value

        :param value:
        :return: None
        """
        max_value = int(''.join(['1' for _ in range(self._bit_width)]), 2)

        if value > max_value:
            raise ValueError('the value {} is larger than '
                             'the maximum value {}'.format(value, max_value))

        self._value = value
        self._text_update()
def set_bit(self, position: int):
        """
        Sets the value at position

        :param position: integer between 0 and 7, inclusive
        :return: None
        """
        if position > (self._bit_width - 1):
            raise ValueError('position greater than the bit width')

        self._value |= (1 << position)
        self._text_update()
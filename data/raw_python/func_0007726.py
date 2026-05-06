def get_bit(self, position: int):
        """
        Returns the bit value at position

        :param position: integer between 0 and <width>, inclusive
        :return: the value at position as a integer
        """

        if position > (self._bit_width - 1):
            raise ValueError('position greater than the bit width')

        if self._value & (1 << position):
            return 1
        else:
            return 0
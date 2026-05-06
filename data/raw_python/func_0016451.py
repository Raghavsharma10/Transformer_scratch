def count(self):
        """Returns the number of bits set to True in the bit string.

        Usage:
            assert BitString('00110').count() == 2

        Arguments: None
        Return:
            An int, the number of bits with value 1.
        """
        result = 0
        bits = self._bits
        while bits:
            result += bits % 2
            bits >>= 1
        return result
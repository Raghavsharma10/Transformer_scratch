def hamming_distance(self, another):
        """
        Compute hamming distance,hamming distance is a total number of different bits of two binary numbers.

        :param another: another simhash value.
        :return: a hamming distance that current simhash and another simhash.
        """
        x = (self.hash ^ another) & ((1 << self.hash_bit_number) - 1)
        result = 0
        while x:
            result += 1
            x &= x - 1
        return result
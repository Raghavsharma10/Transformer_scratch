def is_equal(self, another, limit=0.8):
        """
        Determine two simhash are similar or not similar.

        :param another: another simhash.
        :param limit: a limit of the similarity.
        :return: if similarity greater than limit return true and else return false.
        """
        if another is None:
            raise Exception("Parameter another is null")

        if isinstance(another, int):
            distance = self.hamming_distance(another)
        elif isinstance(another, Simhash):
            assert self.hash_bit_number == another.hash_bit_number
            distance = self.hamming_distance(another.hash)
        else:
            raise Exception("Unsupported parameter type %s" % type(another))

        similarity = float(self.hash_bit_number - distance) / self.hash_bit_number
        if similarity > limit:
            return True
        return False
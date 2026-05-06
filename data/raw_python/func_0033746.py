def generate_random(self, bits_len=None):
        """Generates a random value.

        :param int bits_len:
        :rtype: int
        """
        bits_len = bits_len or self._bits_random
        return random().getrandbits(bits_len)
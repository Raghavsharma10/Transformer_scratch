def cover(cls, bits, wildcard_probability):
        """Create a new bit condition that matches the provided bit string,
        with the indicated per-index wildcard probability.

        Usage:
            condition = BitCondition.cover(bitstring, .33)
            assert condition(bitstring)

        Arguments:
            bits: A BitString which the resulting condition must match.
            wildcard_probability: A float in the range [0, 1] which
            indicates the likelihood of any given bit position containing
            a wildcard.
        Return:
            A randomly generated BitCondition which matches the given bits.
        """

        if not isinstance(bits, BitString):
            bits = BitString(bits)

        mask = BitString([
            random.random() > wildcard_probability
            for _ in range(len(bits))
        ])

        return cls(bits, mask)
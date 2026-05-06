def get_mix_gen(self, sample):
        """Returns function that returns sequence of characters of a
        given length from a given sample
        """
        def mix(length):
            result = "".join(random.choice(sample) for _ in xrange(length)).strip()
            if len(result) == length:
                return result
            return mix(length)
        return mix
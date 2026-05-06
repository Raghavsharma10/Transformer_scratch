def sense(self):
        """Return a situation, encoded as a bit string, which represents
        the observable state of the environment.

        Usage:
            situation = scenario.sense()
            assert isinstance(situation, BitString)

        Arguments: None
        Return:
            The current situation.
        """
        haystack = bitstrings.BitString.random(self.input_size)
        self.needle_value = haystack[self.needle_index]
        return haystack
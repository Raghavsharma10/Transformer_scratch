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
        self.current_situation = bitstrings.BitString([
            random.randrange(2)
            for _ in range(self.address_size + (1 << self.address_size))
        ])
        return self.current_situation
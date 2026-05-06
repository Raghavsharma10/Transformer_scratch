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
        situation = self.wrapped.sense()

        self.logger.debug('Situation: %s', situation)

        return situation
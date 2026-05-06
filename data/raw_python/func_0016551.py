def get_classifications(self):
        """Return the classifications made by the algorithm for this
        scenario.

        Usage:
            model.run(scenario, learn=False)
            classifications = scenario.get_classifications()

        Arguments: None
        Return:
            An indexable sequence containing the classifications made by
            the model for each situation, in the same order as the original
            situations themselves appear.
        """
        if bitstrings.using_numpy():
            return numpy.array(self.classifications)
        else:
            return self.classifications
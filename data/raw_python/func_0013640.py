def getContinuousData(self, referenceName=None, start=None, end=None):
        """
        Returns a set number of simulated continuous data.

        :param referenceName: name of reference to "search" on
        :param start: start coordinate of query
        :param end: end coordinate of query
        :return: Yields continuous list
        """
        randomNumberGenerator = random.Random()
        randomNumberGenerator.seed(self._randomSeed)
        for i in range(100):
            gaContinuous = self._generateSimulatedContinuous(
                                    randomNumberGenerator)
            match = (
                gaContinuous.start < end and
                gaContinuous.end > start and
                gaContinuous.reference_name == referenceName)
            if match:
                yield gaContinuous
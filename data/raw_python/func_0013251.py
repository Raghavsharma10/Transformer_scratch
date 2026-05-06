def getFeature(self, compoundId):
        """
        Fetches a simulated feature by ID.

        :param compoundId: any non-null string
        :return: A simulated feature with id set to the same value as the
            passed-in compoundId.
        ":raises: exceptions.ObjectWithIdNotFoundException if None is passed
            in for the compoundId.
        """
        if compoundId is None:
            raise exceptions.ObjectWithIdNotFoundException(compoundId)
        randomNumberGenerator = random.Random()
        randomNumberGenerator.seed(self._randomSeed)
        feature = self._generateSimulatedFeature(randomNumberGenerator)
        feature.id = str(compoundId)
        feature.parent_id = ""  # TODO: Test with nonempty parentIDs?
        return feature
def getFeatures(self, referenceName=None, start=None, end=None,
                    startIndex=None, maxResults=None,
                    featureTypes=None, parentId=None,
                    name=None, geneSymbol=None, numFeatures=10):
        """
        Returns a set number of simulated features.

        :param referenceName: name of reference to "search" on
        :param start: start coordinate of query
        :param end: end coordinate of query
        :param startIndex: None or int
        :param maxResults: None or int
        :param featureTypes: optional list of ontology terms to limit query
        :param parentId: optional parentId to limit query.
        :param name: the name of the feature
        :param geneSymbol: the symbol for the gene the features are on
        :param numFeatures: number of features to generate in the return.
            10 is a reasonable (if arbitrary) default.
        :return: Yields feature list
        """
        randomNumberGenerator = random.Random()
        randomNumberGenerator.seed(self._randomSeed)
        for featureId in range(numFeatures):
            gaFeature = self._generateSimulatedFeature(randomNumberGenerator)
            gaFeature.id = self.getCompoundIdForFeatureId(featureId)
            match = (
                gaFeature.start < end and
                gaFeature.end > start and
                gaFeature.reference_name == referenceName and (
                    featureTypes is None or len(featureTypes) == 0 or
                    gaFeature.feature_type in featureTypes))
            if match:
                gaFeature.parent_id = ""  # TODO: Test nonempty parentIDs?
                yield gaFeature
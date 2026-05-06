def getFeatures(self, referenceName=None, start=None, end=None,
                    startIndex=None, maxResults=None,
                    featureTypes=None, parentId=None,
                    name=None, geneSymbol=None):
        """
        method passed to runSearchRequest to fulfill the request
        :param str referenceName: name of reference (ex: "chr1")
        :param start: castable to int, start position on reference
        :param end: castable to int, end position on reference
        :param startIndex: none or castable to int
        :param maxResults: none or castable to int
        :param featureTypes: array of str
        :param parentId: none or featureID of parent
        :param name: the name of the feature
        :param geneSymbol: the symbol for the gene the features are on
        :return: yields a protocol.Feature at a time
        """
        with self._db as dataSource:
            features = dataSource.searchFeaturesInDb(
                startIndex, maxResults,
                referenceName=referenceName,
                start=start, end=end,
                parentId=parentId, featureTypes=featureTypes,
                name=name, geneSymbol=geneSymbol)
            for feature in features:
                gaFeature = self._gaFeatureForFeatureDbRecord(feature)
                yield gaFeature
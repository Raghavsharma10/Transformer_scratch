def getFeature(self, compoundId):
        """
        find a feature and return ga4gh representation, use compoundId as
        featureId
        """
        feature = self._getFeatureById(compoundId.featureId)
        feature.id = str(compoundId)
        return feature
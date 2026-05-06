def getCompoundIdForFeatureId(self, featureId):
        """
        Returns server-style compound ID for an internal featureId.

        :param long featureId: id of feature in database
        :return: string representing ID for the specified GA4GH protocol
            Feature object in this FeatureSet.
        """
        if featureId is not None and featureId != "":
            compoundId = datamodel.FeatureCompoundId(
                self.getCompoundId(), str(featureId))
        else:
            compoundId = ""
        return str(compoundId)
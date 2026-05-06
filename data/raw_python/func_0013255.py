def getFeature(self, compoundId):
        """
        Returns a protocol.Feature object corresponding to a compoundId
        :param compoundId: a datamodel.FeatureCompoundId object
        :return: a Feature object.
        :raises: exceptions.ObjectWithIdNotFoundException if invalid
            compoundId is provided.
        """
        featureId = long(compoundId.featureId)
        with self._db as dataSource:
            featureReturned = dataSource.getFeatureById(featureId)

        if featureReturned is None:
            raise exceptions.ObjectWithIdNotFoundException(compoundId)
        else:
            gaFeature = self._gaFeatureForFeatureDbRecord(featureReturned)
            return gaFeature
def removeFeatureSet(self, featureSet):
        """
        Removes the specified featureSet from this repository.
        """
        q = models.Featureset.delete().where(
            models.Featureset.id == featureSet.getId())
        q.execute()
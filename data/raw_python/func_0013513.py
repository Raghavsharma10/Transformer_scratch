def addFeatureSet(self, featureSet):
        """
        Adds the specified featureSet to this dataset.
        """
        id_ = featureSet.getId()
        self._featureSetIdMap[id_] = featureSet
        self._featureSetIds.append(id_)
        name = featureSet.getLocalId()
        self._featureSetNameMap[name] = featureSet
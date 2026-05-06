def addContinuousSet(self, continuousSet):
        """
        Adds the specified continuousSet to this dataset.
        """
        id_ = continuousSet.getId()
        self._continuousSetIdMap[id_] = continuousSet
        self._continuousSetIds.append(id_)
        name = continuousSet.getLocalId()
        self._continuousSetNameMap[name] = continuousSet
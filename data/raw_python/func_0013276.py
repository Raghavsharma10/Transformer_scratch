def addCallSet(self, callSet):
        """
        Adds the specfied CallSet to this VariantSet.
        """
        callSetId = callSet.getId()
        self._callSetIdMap[callSetId] = callSet
        self._callSetNameMap[callSet.getLocalId()] = callSet
        self._callSetIds.append(callSetId)
        self._callSetIdToIndex[callSet.getId()] = len(self._callSetIds) - 1
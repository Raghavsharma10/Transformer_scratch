def addReference(self, reference):
        """
        Adds the specified reference to this ReferenceSet.
        """
        id_ = reference.getId()
        self._referenceIdMap[id_] = reference
        self._referenceNameMap[reference.getLocalId()] = reference
        self._referenceIds.append(id_)
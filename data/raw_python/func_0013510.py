def addVariantSet(self, variantSet):
        """
        Adds the specified variantSet to this dataset.
        """
        id_ = variantSet.getId()
        self._variantSetIdMap[id_] = variantSet
        self._variantSetNameMap[variantSet.getLocalId()] = variantSet
        self._variantSetIds.append(id_)
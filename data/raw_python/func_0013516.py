def addRnaQuantificationSet(self, rnaQuantSet):
        """
        Adds the specified rnaQuantification set to this dataset.
        """
        id_ = rnaQuantSet.getId()
        self._rnaQuantificationSetIdMap[id_] = rnaQuantSet
        self._rnaQuantificationSetIds.append(id_)
        name = rnaQuantSet.getLocalId()
        self._rnaQuantificationSetNameMap[name] = rnaQuantSet
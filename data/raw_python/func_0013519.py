def addPhenotypeAssociationSet(self, phenotypeAssociationSet):
        """
        Adds the specified g2p association set to this backend.
        """
        id_ = phenotypeAssociationSet.getId()
        self._phenotypeAssociationSetIdMap[id_] = phenotypeAssociationSet
        self._phenotypeAssociationSetNameMap[
            phenotypeAssociationSet.getLocalId()] = phenotypeAssociationSet
        self._phenotypeAssociationSetIds.append(id_)
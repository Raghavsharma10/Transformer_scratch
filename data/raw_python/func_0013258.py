def addRnaQuantification(self, rnaQuantification):
        """
        Add an rnaQuantification to this rnaQuantificationSet
        """
        id_ = rnaQuantification.getId()
        self._rnaQuantificationIdMap[id_] = rnaQuantification
        self._rnaQuantificationIds.append(id_)
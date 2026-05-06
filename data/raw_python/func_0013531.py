def getRnaQuantificationSet(self, id_):
        """
        Returns the RnaQuantification set with the specified name, or raises
        a RnaQuantificationSetNotFoundException otherwise.
        """
        if id_ not in self._rnaQuantificationSetIdMap:
            raise exceptions.RnaQuantificationSetNotFoundException(id_)
        return self._rnaQuantificationSetIdMap[id_]
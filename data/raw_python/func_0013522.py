def getContinuousSet(self, id_):
        """
        Returns the ContinuousSet with the specified id, or raises a
        ContinuousSetNotFoundException otherwise.
        """
        if id_ not in self._continuousSetIdMap:
            raise exceptions.ContinuousSetNotFoundException(id_)
        return self._continuousSetIdMap[id_]
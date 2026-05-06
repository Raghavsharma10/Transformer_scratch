def getReadGroup(self, id_):
        """
        Returns the ReadGroup with the specified id if it exists in this
        ReadGroupSet, or raises a ReadGroupNotFoundException otherwise.
        """
        if id_ not in self._readGroupIdMap:
            raise exceptions.ReadGroupNotFoundException(id_)
        return self._readGroupIdMap[id_]
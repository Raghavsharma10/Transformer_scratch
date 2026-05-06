def getReference(self, id_):
        """
        Returns the Reference with the specified ID or raises a
        ReferenceNotFoundException if it does not exist.
        """
        if id_ not in self._referenceIdMap:
            raise exceptions.ReferenceNotFoundException(id_)
        return self._referenceIdMap[id_]